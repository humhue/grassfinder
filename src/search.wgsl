// grassfinder. All hash arithmetic is modulo 2^32, exactly preserving
// the Minecraft bits 16..27 used by the original offset calculation.
struct Plant {
    relative: vec4<i32>,
    observed: vec4<i32>,
};
struct PlantArray { values: array<Plant> };
struct Params {
    base_x: u32, base_z: u32, base_y: u32, count: u32,
    range: i32, y_min: i32, y_count: u32, width: u32,
    plant_count: u32, cutoff: u32, post_112: u32, prune: u32,
    group_size: u32, stage1_count: u32, compact: u32, match_cap: u32,
};
struct SumArray { values: array<u32> };
struct Survivor { idx: u32, total: u32 };
struct SurvivorArray { values: array<Survivor> };
struct Counter { count: atomic<u32> };
struct Match { idx: u32, total: u32 };
// 16-byte header so entries start at an 8-byte aligned address.
struct MatchArray {
    count: atomic<u32>,
    padding: array<u32, 3>,
    values: array<Match>,
};

@group(0) @binding(0) var<storage, read> plants: PlantArray;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> sums: SumArray;
@group(0) @binding(3) var<storage, read_write> survivors: SurvivorArray;
@group(0) @binding(4) var<storage, read_write> survivor_count: Counter;
@group(0) @binding(5) var<storage, read_write> matches: MatchArray;

fn grass_hash(x: u32, y: u32, z: u32) -> u32 {
    let seed = (x * 0x002fc20fu) ^ (z * 0x06ebfff5u) ^ y;
    return seed * seed * 0x0285b825u + seed * 11u;
}

fn position_of(idx: u32) -> vec3<i32> {
    let flat_y = params.base_y + idx;
    let yi = flat_y % params.y_count;
    let dz = flat_y / params.y_count;
    let flat_z = params.base_z + dz;
    let xi = params.base_x + flat_z / params.width;
    let zi = flat_z % params.width;
    return vec3<i32>(i32(xi) - params.range,
                     params.y_min + i32(yi), i32(zi) - params.range);
}

fn grass_difference(candidate: vec3<i32>, plant_index: u32) -> u32 {
    let p = plants.values[plant_index];
    let x = bitcast<u32>(candidate.x) + bitcast<u32>(p.relative.x);
    let z = bitcast<u32>(candidate.z) + bitcast<u32>(p.relative.z);
    var y = bitcast<u32>(candidate.y) + bitcast<u32>(p.relative.y);
    if (params.post_112 != 0u) { y = 0u; }
    let hash = grass_hash(x, y, z);
    let ox = i32((hash >> 16u) & 15u);
    let oy = i32((hash >> 20u) & 15u);
    let oz = i32((hash >> 24u) & 15u);
    return u32(abs(p.observed.x - ox) + abs(p.observed.y - oy) + abs(p.observed.z - oz));
}

fn write_result(idx: u32, total: u32) {
    if (params.compact != 0u) {
        if (total < params.cutoff) {
            let slot = atomicAdd(&matches.count, 1u);
            if (slot < params.match_cap) {
                matches.values[slot] = Match(idx, total);
            }
        }
    } else {
        sums.values[idx] = select(total, 0xffffffffu, total >= params.cutoff);
    }
}

// Single-dispatch baseline (full), per-plant pruning (prune), and grouped pruning.
// For grouped mode, the cutoff is checked only after group_size plants.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let idx = invocation.x;
    if (idx >= params.count) { return; }
    let candidate = position_of(idx);
    var total = 0u;
    if (params.prune == 2u) {
        var i = 0u;
        loop {
            if (i >= params.plant_count) { break; }
            let end = min(i + params.group_size, params.plant_count);
            loop {
                if (i >= end) { break; }
                total += grass_difference(candidate, i);
                i += 1u;
            }
            if (total >= params.cutoff) {
                if (params.compact == 0u) { sums.values[idx] = 0xffffffffu; }
                return;
            }
        }
    } else {
        for (var i = 0u; i < params.plant_count; i += 1u) {
            total += grass_difference(candidate, i);
            if (params.prune == 1u && total >= params.cutoff) {
                if (params.compact == 0u) { sums.values[idx] = 0xffffffffu; }
                return;
            }
        }
    }
    write_result(idx, total);
}

// First stage: evaluate the first K plants, append survivors to a GPU-only list.
// The list is at most params.count long, so there is no survivor overflow.
@compute @workgroup_size(64)
fn stage1(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let idx = invocation.x;
    if (idx >= params.count) { return; }
    let candidate = position_of(idx);
    var total = 0u;
    for (var i = 0u; i < params.stage1_count; i += 1u) {
        total += grass_difference(candidate, i);
    }
    if (total < params.cutoff) {
        let slot = atomicAdd(&survivor_count.count, 1u);
        survivors.values[slot] = Survivor(idx, total);
    } else if (params.compact == 0u) {
        sums.values[idx] = 0xffffffffu;
    }
}

// Second stage is dispatched indirectly using the compact survivor count.
// The stage1 and stage2 passes provide the GPU-side synchronization barrier.
@compute @workgroup_size(64)
fn stage2(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let slot = invocation.x;
    if (slot >= atomicLoad(&survivor_count.count)) { return; }
    let entry = survivors.values[slot];
    let candidate = position_of(entry.idx);
    var total = entry.total;
    for (var i = params.stage1_count; i < params.plant_count; i += 1u) {
        total += grass_difference(candidate, i);
        if (total >= params.cutoff) {
            if (params.compact == 0u) { sums.values[entry.idx] = 0xffffffffu; }
            return;
        }
    }
    write_result(entry.idx, total);
}
