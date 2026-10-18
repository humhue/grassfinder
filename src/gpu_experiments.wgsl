// GPU error lookup table variant. Existing search.wgsl retains the baseline paths.
struct Plant { relative: vec4<i32>, observed: vec4<i32> };
struct PlantArray { values: array<Plant> };
struct Params {
    base_x: u32, base_z: u32, base_y: u32, count: u32,
    range: i32, y_min: i32, y_count: u32, width: u32,
    plant_count: u32, cutoff: u32, post_112: u32, prune: u32,
    group_size: u32, stage1_count: u32, compact: u32, match_cap: u32,
    tile_base: u32, tiles_per_row: u32, y_tiles: u32, halo_x_min: i32,
    halo_x_size: u32, halo_z_min: i32, halo_z_size: u32, halo_y_min: i32,
    halo_y_size: u32, cache_words: u32, reserved0: u32, reserved1: u32,
};
struct SumArray { values: array<u32> };
struct Survivor { idx: u32, total: u32 };
struct SurvivorArray { values: array<Survivor> };
struct Counter { count: atomic<u32> };
struct Match { idx: u32, total: u32 };
struct MatchArray {
    count: atomic<u32>,
    padding: array<u32,3>,
    values: array<Match>,
};
struct ErrorLut { values: array<u32> };
@group(0) @binding(0) var<storage, read> plants: PlantArray;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> sums: SumArray;
@group(0) @binding(3) var<storage, read_write> survivors: SurvivorArray;
@group(0) @binding(4) var<storage, read_write> survivor_count: Counter;
@group(0) @binding(5) var<storage, read_write> matches: MatchArray;
@group(0) @binding(6) var<storage, read> error_lut: ErrorLut;

fn grass_hash(x: u32, y: u32, z: u32) -> u32 {
    let seed = x * 0x002fc20fu ^ z * 0x06ebfff5u ^ y;
    return seed * seed * 0x0285b825u + seed * 11u;
}

fn grass_position(idx: u32) -> vec3<i32> {
    let flat_y = params.base_y + idx;
    let yi = flat_y % params.y_count;
    let dz = flat_y / params.y_count;
    let flat_z = params.base_z + dz;
    let xi = params.base_x + flat_z / params.width;
    let zi = flat_z % params.width;
    return vec3<i32>(i32(xi) - params.range, params.y_min + i32(yi), i32(zi) - params.range);
}

fn lookup_difference(plant_index: u32, packed: u32) -> u32 {
    let value = error_lut.values[plant_index * 1024u + packed / 4u];
    return (value >> ((packed % 4u) * 8u)) & 255u;
}

fn hash_lookup(candidate: vec3<i32>, plant_index: u32) -> u32 {
    let plant = plants.values[plant_index];
    let x = bitcast<u32>(candidate.x) + bitcast<u32>(plant.relative.x);
    let z = bitcast<u32>(candidate.z) + bitcast<u32>(plant.relative.z);
    var y = bitcast<u32>(candidate.y) + bitcast<u32>(plant.relative.y);
    if (params.post_112 != 0u) { y = 0u; }
    let hash = grass_hash(x,y,z);
    return lookup_difference(plant_index, (hash >> 16u) & 4095u);
}

fn write_result(idx: u32, total: u32) {
    if (params.compact != 0u) {
        if (total < params.cutoff) {
            let slot = atomicAdd(&matches.count, 1u);
            if (slot < params.match_cap) {
                matches.values[slot] = Match(idx,total);
            }
        }
    } else {
        sums.values[idx] = select(total,0xffffffffu,total >= params.cutoff);
    }
}

@compute @workgroup_size(64)
fn lut_main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let idx = invocation.x;
    if (idx >= params.count) { return; }
    let candidate = grass_position(idx);
    var sum = 0u;
    for (var i = 0u; i < params.plant_count; i += 1u) {
        sum += hash_lookup(candidate,i);
        if (sum >= params.cutoff) {
            if (params.compact == 0u) { sums.values[idx] = 0xffffffffu; }
            return;
        }
    }
    write_result(idx,sum);
}
