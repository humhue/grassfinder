use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use std::error::Error;
use std::fs;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const WORKGROUP: u32 = 64;
const DEFAULT_CHUNK: u32 = 1_048_576;
const FAIL: u32 = u32::MAX;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Pos { x: i32, y: i32, z: i32 }
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Offset { x: i32, y: i32, z: i32 }
#[derive(Copy, Clone, Debug)]
struct Plant { relative: Pos, observed: Offset }

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Backend { Auto, Gpu, Cpu }
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CpuLoop { Flat, Column, Grouped, Full, Tiled }
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CpuError { Direct, Lut }
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum GpuLoop { Full, Prune, Grouped, TwoStage, Lut }
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum GpuOutput { Full, Compact }

#[derive(Debug)]
struct Options {
    backend: Backend,
    cpu_loop: CpuLoop,
    cpu_error: CpuError,
    tile_size: u32,
    gpu_loop: GpuLoop,
    gpu_output: GpuOutput,
    group_size: u32,
    stage1_plants: u32,
    match_cap: u32,
    range: i32,
    y_min: i32,
    y_max: i32,
    post_112: bool,
    avg_delta: f64,
    chunk: u32,
    file: String,
    sort_plants: bool,
    print_matches: bool,
    verify: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            backend: Backend::Auto, cpu_loop: CpuLoop::Tiled, cpu_error: CpuError::Lut, tile_size: 128, gpu_loop: GpuLoop::Lut,
            gpu_output: GpuOutput::Compact, group_size: 4, stage1_plants: 4, match_cap: 16_384,
            // Defaults match the benchmarked 8-billion-position search.
            range: 10_000, y_min: 62, y_max: 82,
            post_112: false, avg_delta: 5.0, chunk: DEFAULT_CHUNK,
            file: "offsets.txt".into(), sort_plants: true,
            print_matches: true, verify: false,
        }
    }
}

fn help() {
    println!("\
grassfinder - cross-platform GPU/CPU grass-coordinate search

USAGE:
  cargo run --release -- [options]

OPTIONS:
  --backend auto|gpu|cpu   GPU with CPU fallback by default (auto)
  --cpu-loop flat|column|grouped|full|tiled  CPU strategy (default: tiled)
                             flat: old per-position scan (comparison baseline)
                             column: reuse X/Z hashes across Y; prune each plant
                             grouped: column + prune after first four plants
                             full: column + compare all plants, no pruning
                             tiled: cache hashes in overlapping 2D X/Z tiles
  --cpu-error direct|lut  Use arithmetic errors or 4K-entry LUT/plant (default lut)
  --tile-size N           Tiled CPU square edge, 8..128 (default 128)
  --gpu-loop full|prune|grouped|two-stage|lut  (default: lut)
                             lut: per-plant GPU error lookup table with pruning
  --group-size N          Grass tests between checks in grouped mode (default 4)
  --stage1-plants N       Grass tests in stage one (two-stage mode, default 4)
  --gpu-output full|compact   Read back all scores or only matches (default: compact)
  --match-cap N           Compact entries per batch (default 16384; overflow retries in full mode)
  --range N               X and Z in [-N, N), default 10000
  --y-min N               Inclusive minimum Y, default 62
  --y-max N               Exclusive maximum Y, default 82
  --post-1.8              For Minecraft 1.8+: ignore Y in hashing (legacy alias: --post-1.12)
  --delta F               Maximum average error; strict < F, default 5.0
  --chunk N               Positions per GPU dispatch, default 1048576
  --file PATH             Input positions/offsets, default offsets.txt
  --keep-order            Do not sort grass plants by expected error
  --no-matches            Do not print individual matches (benchmarking)
  --verify                Compare first GPU batch with CPU reference
  --help                  Show this help

Input file: six integers per line: observed_x observed_y observed_z offset_x offset_y offset_z.
The FIRST plant establishes the observed coordinate origin (same as humhue).
For normal mode, the supplied search Y is the candidate first-plant world Y.
Results report the candidate first-plant world coordinates and the translation
from screenshot/observed coordinates to world coordinates.

Examples:
  cargo run --release
  cargo run --release -- --range 8 --y-min 62 --y-max 65 --file test_known_offsets.txt --verify
  cargo run --release -- --no-matches
  cargo run --release -- --range 1000 --backend gpu --gpu-loop grouped --group-size 4
  cargo run --release -- --range 1000 --backend gpu --gpu-loop two-stage --stage1-plants 4
  cargo run --release -- --range 1000 --backend gpu --gpu-output compact
  cargo run --release -- --range 1000 --backend cpu
");
}

fn options() -> Result<Option<Options>> {
    let mut opt = Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || -> Result<String> {
            args.next().ok_or_else(|| format!("missing value for {flag}").into())
        };
        match flag.as_str() {
            "--help" | "-h" => { help(); return Ok(None); }
            "--backend" => opt.backend = match value()?.as_str() {
                "auto" => Backend::Auto, "gpu" => Backend::Gpu, "cpu" => Backend::Cpu,
                other => return Err(format!("invalid backend: {other}").into()),
            },
            "--cpu-loop" => opt.cpu_loop = match value()?.as_str() {
                "flat" => CpuLoop::Flat, "column" => CpuLoop::Column,
                "grouped" => CpuLoop::Grouped, "full" => CpuLoop::Full,
                "tiled" => CpuLoop::Tiled,
                other => return Err(format!("invalid CPU loop: {other}").into()),
            },
            "--cpu-error" => opt.cpu_error = match value()?.as_str() {
                "direct" => CpuError::Direct, "lut" => CpuError::Lut,
                other => return Err(format!("invalid CPU error mode: {other}").into()),
            },
            "--tile-size" => opt.tile_size = value()?.parse()?,
            "--gpu-loop" => opt.gpu_loop = match value()?.as_str() {
                "full" => GpuLoop::Full, "prune" => GpuLoop::Prune,
                "grouped" => GpuLoop::Grouped, "two-stage" => GpuLoop::TwoStage,
                "lut" => GpuLoop::Lut,
                other => return Err(format!("invalid GPU loop: {other}").into()),
            },
            "--gpu-output" => opt.gpu_output = match value()?.as_str() {
                "full" => GpuOutput::Full, "compact" => GpuOutput::Compact,
                other => return Err(format!("invalid GPU output: {other}").into()),
            },
            "--group-size" => opt.group_size = value()?.parse()?,
            "--stage1-plants" => opt.stage1_plants = value()?.parse()?,
            "--match-cap" => opt.match_cap = value()?.parse()?,
            "--range" => opt.range = value()?.parse()?,
            "--y-min" => opt.y_min = value()?.parse()?,
            "--y-max" => opt.y_max = value()?.parse()?,
            "--delta" => opt.avg_delta = value()?.parse()?,
            "--chunk" => opt.chunk = value()?.parse()?,
            "--file" => opt.file = value()?,
            "--post-1.8" | "--post-1.12" => opt.post_112 = true,
            "--keep-order" => opt.sort_plants = false,
            "--no-matches" => opt.print_matches = false,
            "--verify" => opt.verify = true,
            other => return Err(format!("unknown option: {other}. Try --help").into()),
        }
    }
    if opt.range <= 0 || opt.range > 100_000_000 {
        return Err("--range must be between 1 and 100000000".into());
    }
    if opt.post_112 { opt.y_min = 0; opt.y_max = 1; }
    if opt.y_max <= opt.y_min || (i64::from(opt.y_max) - i64::from(opt.y_min)) > 1_000_000 {
        return Err("invalid Y bounds".into());
    }
    if !opt.avg_delta.is_finite() || opt.avg_delta <= 0.0 || opt.avg_delta > 45.0 {
        return Err("--delta must be finite and in (0,45]".into());
    }
    if opt.chunk == 0 || opt.chunk > 4_194_240 { // max 65535 workgroups * 64
        return Err("--chunk must be in 1..=4194240".into());
    }
    if opt.group_size == 0 || opt.group_size > 4096 {
        return Err("--group-size must be in 1..=4096".into());
    }
    if !(8..=128).contains(&opt.tile_size) { return Err("--tile-size must be in 8..=128".into()); }
    if opt.stage1_plants == 0 {
        return Err("--stage1-plants must be at least 1".into());
    }
    if opt.match_cap == 0 {
        return Err("--match-cap must be at least 1".into());
    }
    opt.match_cap = opt.match_cap.min(opt.chunk);
    Ok(Some(opt))
}

fn load_plants(path: &str) -> Result<(Pos, Vec<Plant>)> {
    let content = fs::read_to_string(path)?;
    let mut original: Vec<(Pos, Offset)> = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() { continue; }
        let numbers = line.split_whitespace().map(str::parse::<i32>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if numbers.len() != 6 {
            return Err(format!("{path}: line {}: expected six integers", line_no + 1).into());
        }
        let off = Offset { x: numbers[3], y: numbers[4], z: numbers[5] };
        if [off.x, off.y, off.z].iter().any(|&c| !(0..=15).contains(&c)) {
            return Err(format!("{path}: line {}: offsets must be between 0 and 15", line_no + 1).into());
        }
        original.push((Pos { x: numbers[0], y: numbers[1], z: numbers[2] }, off));
    }
    if original.is_empty() { return Err("input file has no grass plants".into()); }
    let origin = original[0].0;
    let plants = original.into_iter().map(|(p, observed)| Plant {
        relative: Pos {
            x: p.x.wrapping_sub(origin.x),
            y: p.y.wrapping_sub(origin.y),
            z: p.z.wrapping_sub(origin.z),
        }, observed,
    }).collect();
    Ok((origin, plants))
}

fn expected_component_error(observed: i32) -> i32 {
    (0..=15).map(|u| (u - observed).abs()).sum::<i32>()
}
fn rank_plants(plants: &mut [Plant]) {
    plants.sort_by_key(|p| std::cmp::Reverse(
        expected_component_error(p.observed.x) +
        expected_component_error(p.observed.y) +
        expected_component_error(p.observed.z)
    ));
}

#[inline(always)]
fn grass_hash32(x: i32, y: i32, z: i32) -> u32 {
    let seed = (x as u32).wrapping_mul(0x2fc20f)
        ^ (z as u32).wrapping_mul(0x6ebfff5) ^ y as u32;
    seed.wrapping_mul(seed).wrapping_mul(0x285b825)
        .wrapping_add(seed.wrapping_mul(11))
}
#[inline(always)]
fn difference(candidate: Pos, plant: Plant, post_112: bool) -> u32 {
    let x = candidate.x.wrapping_add(plant.relative.x);
    let z = candidate.z.wrapping_add(plant.relative.z);
    let y = if post_112 { 0 } else { candidate.y.wrapping_add(plant.relative.y) };
    let hash = grass_hash32(x, y, z);
    (plant.observed.x - ((hash >> 16) & 15) as i32).unsigned_abs()
        + (plant.observed.y - ((hash >> 20) & 15) as i32).unsigned_abs()
        + (plant.observed.z - ((hash >> 24) & 15) as i32).unsigned_abs()
}
fn eval_cpu(candidate: Pos, plants: &[Plant], cutoff: u32, post_112: bool) -> u32 {
    let mut total = 0;
    for &plant in plants {
        total += difference(candidate, plant, post_112);
        if total >= cutoff { return FAIL; }
    }
    total
}


// CPU-only optimization: the X and Z part of the grass seed does not depend on Y.
// Prepare it once per X/Z column and reuse it for every candidate Y.  The GPU
// shader and the CPU reference evaluator above remain unchanged.
#[derive(Copy, Clone)]
struct CpuColumnPlant {
    horizontal_seed: u32,
    relative_y: i32,
    observed: Offset,
}

impl Default for CpuColumnPlant {
    fn default() -> Self {
        Self {
            horizontal_seed: 0,
            relative_y: 0,
            observed: Offset { x: 0, y: 0, z: 0 },
        }
    }
}

#[inline(always)]
fn prepare_cpu_column(x: i32, z: i32, plants: &[Plant], prepared: &mut [CpuColumnPlant]) {
    for (&plant, item) in plants.iter().zip(prepared.iter_mut()) {
        let px = x.wrapping_add(plant.relative.x) as u32;
        let pz = z.wrapping_add(plant.relative.z) as u32;
        *item = CpuColumnPlant {
            horizontal_seed: px.wrapping_mul(0x2fc20f) ^ pz.wrapping_mul(0x6ebfff5),
            relative_y: plant.relative.y,
            observed: plant.observed,
        };
    }
}

#[inline(always)]
fn cpu_prepared_difference(y: i32, plant: CpuColumnPlant, post_112: bool) -> u32 {
    let seed = plant.horizontal_seed ^ (if post_112 {
        0
    } else {
        y.wrapping_add(plant.relative_y) as u32
    });
    let hash = seed.wrapping_mul(seed).wrapping_mul(0x285b825)
        .wrapping_add(seed.wrapping_mul(11));
    (plant.observed.x - ((hash >> 16) & 15) as i32).unsigned_abs()
        + (plant.observed.y - ((hash >> 20) & 15) as i32).unsigned_abs()
        + (plant.observed.z - ((hash >> 24) & 15) as i32).unsigned_abs()
}

#[inline(always)]
fn eval_cpu_prepared(y: i32, prepared: &[CpuColumnPlant], cutoff: u32,
                     post_112: bool, strategy: CpuLoop) -> u32 {
    let mut total = 0;
    match strategy {
        CpuLoop::Grouped => {
            // No data-dependent exit inside the first group.  This may improve
            // instruction scheduling, but can also do more hashes than Column.
            let first = prepared.len().min(4);
            for &plant in &prepared[..first] {
                total += cpu_prepared_difference(y, plant, post_112);
            }
            if total >= cutoff { return FAIL; }
            for &plant in &prepared[first..] {
                total += cpu_prepared_difference(y, plant, post_112);
                if total >= cutoff { return FAIL; }
            }
        }
        CpuLoop::Full => {
            for &plant in prepared {
                total += cpu_prepared_difference(y, plant, post_112);
            }
            if total >= cutoff { return FAIL; }
        }
        CpuLoop::Column | CpuLoop::Flat | CpuLoop::Tiled => {
            for &plant in prepared {
                total += cpu_prepared_difference(y, plant, post_112);
                if total >= cutoff { return FAIL; }
            }
        }
    }
    total
}

// Precompute absolute error for each possible 12-bit grass offset.
// This experiment replaces three subtractions + abs operations per plant
// with one random lookup into a ~44 KiB table (for 11 plants).
fn make_error_tables(plants: &[Plant]) -> Vec<[u8; 4096]> {
    plants.iter().map(|p| {
        let mut table = [0u8; 4096];
        for (packed, item) in table.iter_mut().enumerate() {
            let x = (packed & 15) as i32;
            let y = ((packed >> 4) & 15) as i32;
            let z = ((packed >> 8) & 15) as i32;
            *item = ((p.observed.x-x).abs() + (p.observed.y-y).abs()
                + (p.observed.z-z).abs()) as u8;
        }
        table
    }).collect()
}

#[inline(always)]
fn prepared_lut_difference(y: i32, plant: CpuColumnPlant, table: &[u8;4096],
                           post_112: bool) -> u32 {
    let seed = plant.horizontal_seed ^ if post_112 { 0u32 } else {
        y.wrapping_add(plant.relative_y) as u32
    };
    let hash = seed.wrapping_mul(seed).wrapping_mul(0x285b825)
        .wrapping_add(seed.wrapping_mul(11));
    u32::from(table[((hash >> 16) & 0xfff) as usize])
}

#[inline(always)]
fn eval_cpu_prepared_lut(y: i32, prepared: &[CpuColumnPlant], tables: &[[u8;4096]],
                         cutoff: u32, post_112: bool, strategy: CpuLoop) -> u32 {
    let mut total = 0u32;
    if matches!(strategy, CpuLoop::Grouped) {
        let first = prepared.len().min(4);
        for i in 0..first { total += prepared_lut_difference(y, prepared[i], &tables[i], post_112); }
        if total >= cutoff { return FAIL; }
        for i in first..prepared.len() {
            total += prepared_lut_difference(y, prepared[i], &tables[i], post_112);
            if total >= cutoff { return FAIL; }
        }
    } else if matches!(strategy, CpuLoop::Full) {
        for i in 0..prepared.len() { total += prepared_lut_difference(y, prepared[i], &tables[i], post_112); }
        if total >= cutoff { return FAIL; }
    } else {
        for i in 0..prepared.len() {
            total += prepared_lut_difference(y, prepared[i], &tables[i], post_112);
            if total >= cutoff { return FAIL; }
        }
    }
    total
}

// Divide work by X/Z columns, not individual positions.  This eliminates two
// 64-bit divisions per candidate and reuses two hash multiplications across all
// Y levels.  The first and last columns can be partial, because GPU-style batch
// boundaries need not coincide with a column boundary.
fn scan_cpu_columns(grid: Grid, processed: u64, n: u32, plants: &[Plant],
                    cutoff: u32, post_112: bool, strategy: CpuLoop,
                    tables: Option<&[[u8;4096]]>) -> Vec<(u32, u32)> {
    let y_count = u64::from(grid.y_count);
    let start_col = processed / y_count;
    let end = processed + u64::from(n);
    let end_col = (end - 1) / y_count + 1;
    let columns = end_col - start_col;
    const COLUMNS_PER_BLOCK: u64 = 256;
    let blocks = (columns + COLUMNS_PER_BLOCK - 1) / COLUMNS_PER_BLOCK;

    (0..blocks).into_par_iter()
        .fold(
            || (vec![CpuColumnPlant::default(); plants.len()], Vec::<(u32,u32)>::new()),
            |(mut prepared, mut hits), block| {
                let first_col = start_col + block * COLUMNS_PER_BLOCK;
                let last_col = (first_col + COLUMNS_PER_BLOCK).min(end_col);
                for col in first_col..last_col {
                    // One coordinate decoding per *column*, rather than per Y.
                    let x = (col / u64::from(grid.width)) as i32 - grid.range;
                    let z = (col % u64::from(grid.width)) as i32 - grid.range;
                    prepare_cpu_column(x, z, plants, &mut prepared);
                    let row_start = col * y_count;
                    let y_first = processed.saturating_sub(row_start).min(y_count) as u32;
                    let y_last = end.saturating_sub(row_start).min(y_count) as u32;
                    for yi in y_first..y_last {
                        let y = grid.y_min + yi as i32;
                        let score = if let Some(t) = tables {
                            eval_cpu_prepared_lut(y, &prepared, t, cutoff, post_112, strategy)
                        } else { eval_cpu_prepared(y, &prepared, cutoff, post_112, strategy) };
                        if score != FAIL {
                            let local = (row_start + u64::from(yi) - processed) as u32;
                            hits.push((local, score));
                        }
                    }
                }
                (prepared, hits)
            },
        )
        .map(|(_, hits)| hits)
        .reduce(Vec::new, |mut a, mut b| { a.append(&mut b); a })
}

// CPU-only spatial tile cache.  Every tile computes each relevant world
// grass hash once, then the candidate/pruning loop retrieves the packed 12-bit
// offset.  Halo dimensions cover all relative plant positions.  No GPU change.
#[derive(Clone, Copy)]
struct TileHalo { min_x: i32, min_y: i32, min_z: i32, nx: usize, ny: usize, nz: usize }
fn tile_halo(plants: &[Plant], post_112: bool, tile_size: u32, y_count: u32) -> Result<TileHalo> {
    let xs = plants.iter().map(|p| p.relative.x);
    let ys = plants.iter().map(|p| if post_112 {0} else {p.relative.y});
    let zs = plants.iter().map(|p| p.relative.z);
    let min_x = xs.clone().min().unwrap_or(0);
    let min_y = ys.clone().min().unwrap_or(0);
    let min_z = zs.clone().min().unwrap_or(0);
    let dx = i64::from(xs.max().unwrap_or(0)) - i64::from(min_x);
    let dy = i64::from(ys.max().unwrap_or(0)) - i64::from(min_y);
    let dz = i64::from(zs.max().unwrap_or(0)) - i64::from(min_z);
    let nx = usize::try_from(i64::from(tile_size) + dx)?;
    let ny = usize::try_from(i64::from(y_count) + dy)?;
    let nz = usize::try_from(i64::from(tile_size) + dz)?;
    let n = nx.checked_mul(ny).and_then(|v| v.checked_mul(nz)).ok_or("tile too large")?;
    if n > 4_000_000 { return Err(format!("spatial cache too large: {n} u16 entries per Rayon worker; reduce --tile-size or plant coordinate spread").into()); }
    Ok(TileHalo { min_x, min_y, min_z, nx, ny, nz })
}

#[inline(always)]
fn direct_packed_error(packed: u16, observed: Offset) -> u32 {
    (observed.x - i32::from(packed & 15)).unsigned_abs()
      + (observed.y - i32::from((packed >> 4) & 15)).unsigned_abs()
      + (observed.z - i32::from((packed >> 8) & 15)).unsigned_abs()
}

#[inline(always)]
fn eval_cached<const USE_LUT: bool>(base: usize, cache: &[u16], deltas: &[usize],
                                    plants: &[Plant], tables: &[[u8;4096]], cutoff: u32) -> u32 {
    let mut total = 0u32;
    let first = plants.len().min(4);
    for i in 0..first {
        let h = cache[base + deltas[i]];
        total += if USE_LUT { u32::from(tables[i][h as usize]) }
            else { direct_packed_error(h, plants[i].observed) };
    }
    if total >= cutoff { return FAIL; }
    for i in first..plants.len() {
        let h = cache[base + deltas[i]];
        total += if USE_LUT { u32::from(tables[i][h as usize]) }
            else { direct_packed_error(h, plants[i].observed) };
        if total >= cutoff { return FAIL; }
    }
    total
}

// Tiles are independent, with no cache shared across tiles.  They are evaluated
// in batches of 64; each worker reuses its own cache allocation.
fn scan_cpu_tile_batch<const USE_LUT: bool>(grid: Grid, tile_size: u32, begin: u64, end: u64,
        plants: &[Plant], tables: &[[u8;4096]], cutoff: u32, post_112: bool,
        halo: TileHalo) -> (u64, Vec<(Pos,u32)>) {
    let ntz = u64::from(grid.width.div_ceil(tile_size));
    (begin..end).into_par_iter().fold(
        || (Vec::<u16>::new(), 0u64, Vec::<(Pos,u32)>::new()),
        |(mut cache, mut count, mut hits), id| {
            let tx = (id / ntz) as u32;
            let tz = (id % ntz) as u32;
            let x_start = tx * tile_size;
            let z_start = tz * tile_size;
            let cx = (grid.width-x_start).min(tile_size) as usize;
            let cz = (grid.width-z_start).min(tile_size) as usize;
            let sx = halo.nx - tile_size as usize + cx;
            let sz = halo.nz - tile_size as usize + cz;
            let sy = halo.ny;
            let n = sx*sz*sy;
            cache.resize(n, 0);
            let stride_x = sz*sy;
            let stride_z = sy;
            let base_x = x_start as i32 - grid.range;
            let base_z = z_start as i32 - grid.range;
            for ix in 0..sx {
                let wx = base_x.wrapping_add(halo.min_x).wrapping_add(ix as i32);
                let a = (wx as u32).wrapping_mul(0x2fc20f);
                for iz in 0..sz {
                    let wz = base_z.wrapping_add(halo.min_z).wrapping_add(iz as i32);
                    let horiz = a ^ (wz as u32).wrapping_mul(0x6ebfff5);
                    let off = ix*stride_x + iz*stride_z;
                    for iy in 0..sy {
                        let wy = grid.y_min.wrapping_add(halo.min_y).wrapping_add(iy as i32);
                        let seed = horiz ^ if post_112 { 0 } else { wy as u32 };
                        let h = seed.wrapping_mul(seed).wrapping_mul(0x285b825)
                            .wrapping_add(seed.wrapping_mul(11));
                        cache[off+iy] = ((h >> 16) & 0xfff) as u16;
                    }
                }
            }
            let deltas: Vec<usize> = plants.iter().map(|p| {
                let dx = (i64::from(p.relative.x)-i64::from(halo.min_x)) as usize;
                let dy = (if post_112 {0} else {i64::from(p.relative.y)})-i64::from(halo.min_y);
                let dz = (i64::from(p.relative.z)-i64::from(halo.min_z)) as usize;
                dx*stride_x + dz*stride_z + (dy as usize)
            }).collect();
            count += (cx*cz) as u64 * u64::from(grid.y_count);
            for ix in 0..cx {
                for iz in 0..cz {
                    let base = ix*stride_x + iz*stride_z;
                    for iy in 0..(grid.y_count as usize) {
                        let score = eval_cached::<USE_LUT>(base+iy, &cache, &deltas, plants, tables, cutoff);
                        if score != FAIL {
                            hits.push((Pos { x: base_x+ix as i32, y: grid.y_min+iy as i32,
                                z:base_z+iz as i32 }, score));
                        }
                    }
                }
            }
            (cache, count, hits)
        }).map(|(_,n,h)| (n,h))
        .reduce(|| (0u64, Vec::<(Pos,u32)>::new()), |(a,mut ah),(b,mut bh)| {
            ah.append(&mut bh); (a+b,ah)
        })
}

fn run_cpu_tiles(grid: Grid, opt: &Options, plants: &[Plant], tables: &[[u8;4096]],
                 cutoff: u32, origin: Pos) -> Result<()> {
    let halo = tile_halo(plants, opt.post_112, opt.tile_size, grid.y_count)?;
    let tile_dim = u64::from(grid.width.div_ceil(opt.tile_size));
    let n_tiles = tile_dim * tile_dim;
    eprintln!("Tile cache: {}x{} X/Z; packed u16 hashes; max {:.2} MiB/worker",
        opt.tile_size, opt.tile_size,
        (halo.nx*halo.ny*halo.nz*2) as f64 / (1024.0*1024.0));
    let start = Instant::now();
    let mut last_report = Instant::now();
    let mut done = 0u64;
    let mut matched = 0u64;
    for first in (0..n_tiles).step_by(64) {
        let last = (first+64).min(n_tiles);
        let (count, matches) = if opt.cpu_error == CpuError::Lut {
            scan_cpu_tile_batch::<true>(grid, opt.tile_size, first, last, plants, tables, cutoff,
                opt.post_112, halo)
        } else {
            scan_cpu_tile_batch::<false>(grid, opt.tile_size, first, last, plants, tables, cutoff,
                opt.post_112, halo)
        };
        done += count;
        matched += matches.len() as u64;
        if opt.print_matches {
            for (pos, score) in matches { report_match(pos, origin, score, plants.len()); }
        }
        if last_report.elapsed() >= Duration::from_secs(5) || last == n_tiles {
            progress(done, grid.total, start, matched);
            last_report = Instant::now();
        }
    }
    eprintln!("Finished: {} candidates, {} matches, {:.3} seconds, {:.3} million candidates/second.",
        done, matched, start.elapsed().as_secs_f64(),
        done as f64 / start.elapsed().as_secs_f64().max(0.000001) / 1_000_000.0);
    Ok(())
}

#[derive(Copy, Clone, Debug)]
struct Grid { range: i32, width: u32, y_min: i32, y_count: u32, total: u64 }
impl Grid {
    fn new(options: &Options) -> Result<Self> {
        let width = u32::try_from(options.range * 2)?;
        let y_count = u32::try_from(i64::from(options.y_max) - i64::from(options.y_min))?;
        let total = u64::from(width).checked_mul(u64::from(width))
            .and_then(|n| n.checked_mul(u64::from(y_count)))
            .ok_or("search space too large for u64 candidate index")?;
        Ok(Self { range: options.range, width, y_min: options.y_min, y_count, total })
    }
    fn components(&self, index: u64) -> (u32,u32,u32) {
        let y = (index % u64::from(self.y_count)) as u32;
        let yz = index / u64::from(self.y_count);
        let z = (yz % u64::from(self.width)) as u32;
        let x = (yz / u64::from(self.width)) as u32;
        (x, z, y)
    }
    fn position(&self, index: u64) -> Pos {
        let (x,z,y) = self.components(index);
        Pos { x: x as i32 - self.range, y: self.y_min + y as i32, z: z as i32 - self.range }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuPlant { relative: [i32;4], observed: [i32;4] }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParams {
    base_x:u32, base_z:u32, base_y:u32, count:u32,
    range:i32, y_min:i32, y_count:u32, width:u32,
    plant_count:u32, cutoff:u32, post_112:u32, prune:u32,
    group_size:u32, stage1_count:u32, compact:u32, match_cap:u32,
    tile_base:u32, tiles_per_row:u32, y_tiles:u32, halo_x_min:i32,
    halo_x_size:u32, halo_z_min:i32, halo_z_size:u32, halo_y_min:i32,
    halo_y_size:u32, cache_words:u32, reserved0:u32, reserved1:u32,
}

struct GpuSearch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    stage1_pipeline: wgpu::ComputePipeline,
    stage2_pipeline: wgpu::ComputePipeline,
    lut_pipeline: wgpu::ComputePipeline,
    experimental_bind_group: wgpu::BindGroup,
    indirect_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    indirect_bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
    survivor_counter: wgpu::Buffer,
    dispatch_args: wgpu::Buffer,
    match_buffer: wgpu::Buffer,
    compact_staging: wgpu::Buffer,
    chunk: u32,
    match_cap: u32,
}
impl GpuSearch {
    async fn new(plants: &[Plant], chunk: u32, match_cap: u32) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }).await.map_err(|e| format!("GPU adapter unavailable: {e}"))?;
        let info = adapter.get_info();
        eprintln!("GPU adapter: {} | {:?} | {:?}", info.name, info.backend, info.device_type);
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("grassfinder"),
            ..Default::default()
        }).await.map_err(|e| format!("could not create GPU device: {e}"))?;
        if chunk.div_ceil(WORKGROUP) > device.limits().max_compute_workgroups_per_dimension {
            return Err(format!("--chunk too large for this GPU (max {} positions)",
                device.limits().max_compute_workgroups_per_dimension * WORKGROUP).into());
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grass search"),
            source: wgpu::ShaderSource::Wgsl(include_str!("search.wgsl").into()),
        });
        let entries = (0..=5).map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 1 { wgpu::BufferBindingType::Uniform } else {
                    wgpu::BufferBindingType::Storage { read_only: binding == 0 }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }).collect::<Vec<_>>();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grass search layout"), entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grass search pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mk_pipeline = |label: &'static str, entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: Some(&pipeline_layout), module: &shader,
                entry_point: Some(entry), compilation_options: Default::default(), cache: None,
            })
        };
        let pipeline = mk_pipeline("grass single-dispatch", "main");
        let stage1_pipeline = mk_pipeline("grass survivor filter", "stage1");
        let stage2_pipeline = mk_pipeline("grass survivor completion", "stage2");

        let indirect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grass indirect dispatch preparation"),
            source: wgpu::ShaderSource::Wgsl(include_str!("dispatch.wgsl").into()),
        });
        let indirect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prepare grass stage 2 dispatch"), layout: None,
            module: &indirect_shader, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });
        let gpu_plants: Vec<_> = plants.iter().map(|p| GpuPlant {
            relative: [p.relative.x, p.relative.y, p.relative.z, 0],
            observed: [p.observed.x, p.observed.y, p.observed.z, 0],
        }).collect();
        let plants_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grass plants"),
            contents: bytemuck::cast_slice(&gpu_plants),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // Four packed u8 errors per u32: 11 plants * 1024 words = 44 KiB.
        let error_tables = make_error_tables(plants);
        let mut packed_lut: Vec<u32> = Vec::with_capacity(plants.len() * 1024);
        for table in &error_tables {
            for bytes in table.chunks_exact(4) {
                packed_lut.push(u32::from_le_bytes(bytes.try_into().unwrap()));
            }
        }
        let lut_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU packed plant error LUT"),
            contents: bytemuck::cast_slice(&packed_lut),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("search chunk parameters"), size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_size = u64::from(chunk) * 4;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU candidate deltas"), size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CPU readback of all deltas"), size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let survivors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact stage 1 survivors"), size: u64::from(chunk) * 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let survivor_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stage 1 survivor counter"), size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dispatch_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stage 2 indirect dispatch arguments"), size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let compact_bytes = 16 + u64::from(match_cap) * 8;
        let match_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact final matches"), size: compact_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let compact_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CPU readback of compact matches"), size: compact_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grass search inputs"), layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: plants_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: survivors.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: survivor_counter.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: match_buffer.as_entire_binding() },
            ],
        });
        // Independent pipeline layout so the original GPU shader is untouched.
        let mut exp_entries = entries.clone();
        exp_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false, min_binding_size: None,
            }, count: None,
        });
        let exp_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grass GPU LUT layout"), entries: &exp_entries,
        });
        let exp_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grass GPU LUT pipeline layout"),
            bind_group_layouts: &[Some(&exp_layout)], immediate_size: 0,
        });
        let exp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grass GPU packed error lookup shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_experiments.wgsl").into()),
        });
        let make_exp_pipeline = |label: &'static str, entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: Some(&exp_pipeline_layout), module: &exp_shader,
                entry_point: Some(entry), compilation_options: Default::default(), cache: None,
            })
        };
        let lut_pipeline = make_exp_pipeline("grass GPU error LUT", "lut_main");
        let experimental_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grass GPU LUT inputs"), layout: &exp_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: plants_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: survivors.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: survivor_counter.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: match_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: lut_buffer.as_entire_binding() },
            ],
        });
        let indirect_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prepare indirect dispatch"),
            layout: &indirect_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {binding: 0, resource: survivor_counter.as_entire_binding()},
                wgpu::BindGroupEntry {binding: 1, resource: dispatch_args.as_entire_binding()},
            ],
        });
        Ok(Self { device, queue, pipeline, stage1_pipeline, stage2_pipeline,
            lut_pipeline, experimental_bind_group,
            indirect_pipeline, bind_group, indirect_bind_group, params_buffer,
            output, staging, survivor_counter, dispatch_args, match_buffer,
            compact_staging, chunk, match_cap })
    }
    fn process(&mut self, grid: Grid, base: u64, n: u32,
        count_plants: u32, cutoff: u32, post_112: bool, loop_mode: GpuLoop,
        group_size: u32, stage1_count: u32, output_mode: GpuOutput,
        mut on_result: impl FnMut(u32, u32),
    ) -> Result<bool> {
        debug_assert!(n <= self.chunk);
        let (base_x,base_z,base_y) = grid.components(base);
        let params = GpuParams {
            base_x, base_z, base_y, count: n, range: grid.range,
            y_min: grid.y_min, y_count: grid.y_count, width: grid.width,
            plant_count: count_plants, cutoff,
            post_112: u32::from(post_112),
            prune: match loop_mode { GpuLoop::Full => 0, GpuLoop::Prune => 1,
                GpuLoop::Grouped => 2, GpuLoop::TwoStage => 1,
                GpuLoop::Lut => 1 },
            group_size, stage1_count, compact: u32::from(output_mode == GpuOutput::Compact),
            match_cap: self.match_cap,
            tile_base:0, tiles_per_row:0, y_tiles:0, halo_x_min:0,
            halo_x_size:0, halo_z_min:0, halo_z_size:0, halo_y_min:0,
            halo_y_size:0, cache_words:0, reserved0:0, reserved1:0,
        };
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("grass search command encoder"),
        });
        if output_mode == GpuOutput::Compact {
            encoder.clear_buffer(&self.match_buffer, 0, Some(4));
        }
        if loop_mode == GpuLoop::TwoStage {
            encoder.clear_buffer(&self.survivor_counter, 0, Some(4));
            {
                let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("grass stage 1"), timestamp_writes: None,
                });
                compute.set_pipeline(&self.stage1_pipeline);
                compute.set_bind_group(0, &self.bind_group, &[]);
                compute.dispatch_workgroups(n.div_ceil(WORKGROUP), 1, 1);
            }
            {
                let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("grass prepare stage 2 indirect dispatch"), timestamp_writes: None,
                });
                compute.set_pipeline(&self.indirect_pipeline);
                compute.set_bind_group(0, &self.indirect_bind_group, &[]);
                compute.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("grass stage 2"), timestamp_writes: None,
                });
                compute.set_pipeline(&self.stage2_pipeline);
                compute.set_bind_group(0, &self.bind_group, &[]);
                compute.dispatch_workgroups_indirect(&self.dispatch_args, 0);
            }
        } else {
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grass single-dispatch pass"), timestamp_writes: None,
            });
            if loop_mode == GpuLoop::Lut {
                compute.set_pipeline(&self.lut_pipeline);
                compute.set_bind_group(0, &self.experimental_bind_group, &[]);
            } else {
                compute.set_pipeline(&self.pipeline);
                compute.set_bind_group(0, &self.bind_group, &[]);
            }
            compute.dispatch_workgroups(n.div_ceil(WORKGROUP), 1, 1);
        }
        let (buffer,bytes) = if output_mode == GpuOutput::Compact {
            let bytes = 16 + u64::from(self.match_cap) * 8;
            encoder.copy_buffer_to_buffer(&self.match_buffer, 0, &self.compact_staging, 0, bytes);
            (&self.compact_staging, bytes)
        } else {
            let bytes = u64::from(n) * 4;
            encoder.copy_buffer_to_buffer(&self.output, 0, &self.staging, 0, bytes);
            (&self.staging, bytes)
        };
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = buffer.slice(0..bytes);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })?;
        rx.recv()??;
        let mut validation_error: Option<String> = None;
        let mut needs_full_retry = false;
        {
            let view = slice.get_mapped_range()?;
            if output_mode == GpuOutput::Compact {
                let count = u32::from_le_bytes(view[0..4].try_into()?);
                if count > self.match_cap {
                    // We have not delivered any callbacks for this batch. Retry it
                    // with full-score readback, preserving every match exactly.
                    needs_full_retry = true;
                } else {
                    for entry in view[16..16 + count as usize * 8].chunks_exact(8) {
                        let idx = u32::from_le_bytes(entry[0..4].try_into()?);
                        let delta = u32::from_le_bytes(entry[4..8].try_into()?);
                        if idx >= n || delta >= cutoff {
                            validation_error = Some(format!("invalid compact GPU result: local index={idx} delta={delta}"));
                            break;
                        }
                        on_result(idx,delta);
                    }
                }
            } else {
                for (idx, b) in view.chunks_exact(4).enumerate() {
                    let delta = u32::from_le_bytes([b[0],b[1],b[2],b[3]]);
                    // V2 verifies rejected candidates by initializing its verification
                    // vector to FAIL. No need to invoke the result callback for them.
                    if delta != FAIL {
                        on_result(idx as u32, delta);
                    }
                }
            }
        }
        // Dropping the mapped view before unmap is required by wgpu.
        buffer.unmap();
        if let Some(err) = validation_error { return Err(err.into()); }
        Ok(!needs_full_retry)
    }
}

fn report_match(candidate: Pos, origin: Pos, sum: u32, n_plants: usize) {
    println!("{:>8}{:>8}{:>8} has an average grass delta of {:.3}, delta between positions is {:>8}{:>8}{:>8}",
        candidate.x, candidate.y, candidate.z, sum as f64 / n_plants as f64,
        candidate.x.wrapping_sub(origin.x), candidate.y.wrapping_sub(origin.y),
        candidate.z.wrapping_sub(origin.z));
}

fn progress(done: u64, total: u64, since: Instant, matches: u64) {
    let elapsed = since.elapsed().as_secs_f64();
    let rate = done as f64 / elapsed.max(0.000001);
    eprintln!("{:.2}% | {:>12}/{:<12} positions | {:.2} M positions/s | matches: {} | {:.1}s",
        100.0 * done as f64 / total as f64,
        done, total, rate / 1_000_000.0, matches, elapsed);
}

fn main() -> Result<()> {
    let opt = match options()? { Some(v) => v, None => return Ok(()) };
    let (origin, mut plants) = load_plants(&opt.file)?;
    if opt.gpu_loop == GpuLoop::TwoStage && opt.stage1_plants > plants.len() as u32 {
        return Err(format!("--stage1-plants {} exceeds plant count {}", opt.stage1_plants, plants.len()).into());
    }
    if opt.sort_plants { rank_plants(&mut plants); }
    if plants.len() > (u32::MAX as usize / 45) {
        return Err("too many plants for 32-bit accumulated error".into());
    }
    let cutoff = (opt.avg_delta * plants.len() as f64).ceil() as u32;
    let grid = Grid::new(&opt)?;
    eprintln!("Plants: {} (first observed plant at {}, {}, {}) | avg delta < {} | total delta < {}",
        plants.len(), origin.x,origin.y,origin.z,opt.avg_delta,cutoff);
    eprintln!("Search: X/Z [-{}, {}), Y [{}, {}) | {} candidates | batch {} | sorted {}",
        grid.range, grid.range, grid.y_min, grid.y_min + grid.y_count as i32,
        grid.total, opt.chunk, opt.sort_plants);

    let mut gpu = if opt.backend != Backend::Cpu {
        match pollster::block_on(GpuSearch::new(&plants, opt.chunk, opt.match_cap)) {
            Ok(g) => Some(g),
            Err(e) if opt.backend == Backend::Auto => {
                eprintln!("GPU unavailable ({e}); using Rayon CPU fallback.");
                None
            }
            Err(e) => return Err(e),
        }
    } else { None };
    eprintln!("Mode: {}", if gpu.is_some() {"GPU"} else {"CPU (Rayon)"});
    if gpu.is_none() { eprintln!("CPU loop: {:?} | Rayon workers: {} | error: {:?}",
        opt.cpu_loop, rayon::current_num_threads(), opt.cpu_error); }
    let error_tables = if gpu.is_none() { make_error_tables(&plants) } else { Vec::new() };
    if gpu.is_none() && opt.cpu_loop == CpuLoop::Tiled {
        return run_cpu_tiles(grid, &opt, &plants, &error_tables, cutoff, origin);
    }
    if gpu.is_some() { eprintln!("GPU loop: {:?} | output: {:?} | group-size: {} | first-stage plants: {} | match-cap: {}",
        opt.gpu_loop, opt.gpu_output, opt.group_size, opt.stage1_plants, opt.match_cap); }
    let mut processed: u64 = 0;
    let mut matched: u64 = 0;
    let mut compact_retries: u64 = 0;
    let mut last_report = Instant::now();
    let start = Instant::now();
    while processed < grid.total {
        let n = (grid.total - processed).min(u64::from(opt.chunk)) as u32;
        if let Some(ref mut gpu) = gpu {
            // Optional CPU/GPU cross-check on first batch only, including rejected candidates.
            let validate = if opt.verify && processed == 0 {
                Some((0..n.min(65_536)).map(|local| eval_cpu(grid.position(processed + u64::from(local)),
                    &plants, cutoff, opt.post_112)).collect::<Vec<_>>())
            } else { None };
            // With compact GPU output, only matches are returned, not rejected entries.
            // Reconstruct a FAIL-filled reference vector to verify the exact same criterion.
            let mut verified_scores = validate.as_ref().map(|v| vec![FAIL; v.len()]);
            let mut on_result = |idx: u32, delta: u32| {
                if let Some(ref mut actual) = verified_scores {
                    if (idx as usize) < actual.len() {
                        actual[idx as usize] = delta;
                    }
                }
                if delta != FAIL {
                    matched += 1;
                    if opt.print_matches {
                        report_match(grid.position(processed + u64::from(idx)), origin, delta, plants.len());
                    }
                }
            };
            let complete = gpu.process(grid, processed, n, plants.len() as u32, cutoff, opt.post_112,
                opt.gpu_loop, opt.group_size, opt.stage1_plants, opt.gpu_output,
                &mut on_result)?;
            if !complete {
                // Compact output overflowed. It reported zero callbacks for this
                // batch, so retrying with full output cannot duplicate matches.
                compact_retries += 1;
                if compact_retries <= 3 {
                    eprintln!("Compact result capacity exceeded in batch starting at {processed}; retrying this batch with full readback.");
                }
                let completed = gpu.process(grid, processed, n, plants.len() as u32, cutoff,
                    opt.post_112, opt.gpu_loop, opt.group_size, opt.stage1_plants,
                    GpuOutput::Full, &mut on_result)?;
                if !completed { return Err("full-output GPU retry unexpectedly incomplete".into()); }
            }
            drop(on_result);
            if let (Some(expected), Some(actual)) = (validate, verified_scores) {
                if let Some((idx, (&cpu, &gpu))) = expected.iter().zip(actual.iter()).enumerate()
                    .find(|(_, (a,b))| a != b) {
                    return Err(format!("GPU/CPU mismatch at index {} (candidate {:?}): GPU={}, CPU={}",
                        processed + idx as u64, grid.position(processed + idx as u64), gpu, cpu).into());
                }
                eprintln!("Verified first {} candidates against CPU.", n.min(65_536));
            }
        } else {
            let matches: Vec<(u32,u32)> = if opt.cpu_loop == CpuLoop::Flat {
                // Unchanged CPU baseline for A/B benchmarking.
                (0..n).into_par_iter().filter_map(|local| {
                    let pos = grid.position(processed + u64::from(local));
                    let sum = eval_cpu(pos, &plants, cutoff, opt.post_112);
                    (sum != FAIL).then_some((local, sum))
                }).collect()
            } else {
                scan_cpu_columns(grid, processed, n, &plants, cutoff, opt.post_112, opt.cpu_loop,
                    (opt.cpu_error == CpuError::Lut).then_some(error_tables.as_slice()))
            };
            matched += matches.len() as u64;
            if opt.print_matches {
                for (local, delta) in matches {
                    report_match(grid.position(processed + u64::from(local)), origin, delta, plants.len());
                }
            }
        }
        processed += u64::from(n);
        if last_report.elapsed() >= Duration::from_secs(5) || processed == grid.total {
            progress(processed, grid.total, start, matched);
            last_report = Instant::now();
        }
    }
    if compact_retries > 0 {
        eprintln!("Compact output overflowed in {compact_retries} batches; all were retried with full-output readback. Consider increasing --match-cap for this threshold.");
    }
    eprintln!("Finished: {} candidates, {} matches, {:.3} seconds, {:.3} million candidates/second.",
        processed, matched, start.elapsed().as_secs_f64(),
        processed as f64 / start.elapsed().as_secs_f64().max(0.000001) / 1_000_000.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    // Reference: same wrapping semantics as the original humhue hash on i32 inputs.
    fn hash64_reference(x:i32,y:i32,z:i32) -> u32 {
        let seed = i64::from(x.wrapping_mul(0x2fc20f))
            ^ i64::from(z.wrapping_mul(0x6ebfff5)) ^ i64::from(y);
        let seed = seed.wrapping_mul(seed).wrapping_mul(0x285b825)
            .wrapping_add(seed.wrapping_mul(11));
        seed as u32
    }
    #[test]
    fn lower_32_bits_agree_with_original() {
        let mut s:u32 = 0x12345678;
        for _ in 0..300_000 {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            let x = s as i32;
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            let y = s as i32;
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            let z = s as i32;
            assert_eq!(grass_hash32(x,y,z), hash64_reference(x,y,z));
        }
    }
    #[test]
    fn optimized_defaults() {
        let defaults = Options::default();
        assert_eq!(defaults.backend, Backend::Auto);
        assert_eq!(defaults.cpu_loop, CpuLoop::Tiled);
        assert_eq!(defaults.cpu_error, CpuError::Lut);
        assert_eq!(defaults.tile_size, 128);
        assert_eq!(defaults.gpu_loop, GpuLoop::Lut);
        assert_eq!(defaults.gpu_output, GpuOutput::Compact);
        assert_eq!(defaults.range, 10_000);
        assert_eq!((defaults.y_min, defaults.y_max), (62, 82));
        assert_eq!(defaults.avg_delta, 5.0);
        assert!(defaults.sort_plants);
        assert_eq!(Grid::new(&defaults).unwrap().total, 8_000_000_000);
    }
    #[test]
    fn origin_and_offsets() {
        let file = include_str!("../offsets.txt");
        assert_eq!(file.lines().count(), 11);
    }
    #[test]
    fn linear_grid_mapping() {
        let opt = Options {range: 5,y_min:62,y_max:65,..Default::default()};
        let grid = Grid::new(&opt).unwrap();
        assert_eq!(grid.total, 300);
        assert_eq!(grid.position(0), Pos{x:-5,y:62,z:-5});
        assert_eq!(grid.position(2), Pos{x:-5,y:64,z:-5});
        assert_eq!(grid.position(3), Pos{x:-5,y:62,z:-4});
        assert_eq!(grid.position(299), Pos{x:4,y:64,z:4});
    }
    #[test]
    fn synthetic_fixture_contains_exact_match() {
        let (_origin, plants) = load_plants("test_known_offsets.txt").unwrap();
        let candidate = Pos { x: 0, y: 63, z: 0 };
        assert_eq!(eval_cpu(candidate, &plants, 1, false), 0);
    }

    #[test]
    fn cpu_column_hash_matches_reference_for_all_strategies() {
        let (_, mut plants) = load_plants("offsets.txt").unwrap();
        rank_plants(&mut plants);
        let mut prepared = vec![CpuColumnPlant::default(); plants.len()];
        for &post in &[false, true] {
            for x in [-10000, -1836, 0, 1414, 9999] {
                for z in [-10000, -451, 0, 1414, 9999] {
                    prepare_cpu_column(x, z, &plants, &mut prepared);
                    for y in 62..127 {
                        let pos = Pos {x,y,z};
                        for cutoff in [1, 22, 55, 121, 496] {
                            let expected = eval_cpu(pos, &plants, cutoff, post);
                            for strategy in [CpuLoop::Column, CpuLoop::Grouped, CpuLoop::Full] {
                                assert_eq!(eval_cpu_prepared(y, &prepared, cutoff, post, strategy),
                                           expected, "candidate {pos:?}, strategy {strategy:?}, post {post}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cpu_column_batches_match_flat_scan_even_on_partial_columns() {
        let (_, mut plants) = load_plants("offsets.txt").unwrap();
        rank_plants(&mut plants);
        let opt = Options {range: 8, y_min: 62, y_max: 69, ..Default::default()};
        let grid = Grid::new(&opt).unwrap();
        for (processed, n) in [(0u64, 5u32), (3, 41), (48, 102), (300, 127), (grid.total-4, 4)] {
            for cutoff in [22, 55, 121, 496] {
                let mut expected: Vec<(u32,u32)> = (0..n).filter_map(|local| {
                    let sum = eval_cpu(grid.position(processed + u64::from(local)), &plants, cutoff, false);
                    (sum != FAIL).then_some((local, sum))
                }).collect();
                expected.sort_unstable();
                for strategy in [CpuLoop::Column, CpuLoop::Grouped, CpuLoop::Full] {
                    let mut actual = scan_cpu_columns(grid, processed, n, &plants, cutoff, false, strategy, None);
                    actual.sort_unstable();
                    assert_eq!(actual, expected, "start {processed}, strategy {strategy:?}, cutoff {cutoff}");
                }
            }
        }
    }

    #[test]
    fn cpu_lut_and_tile_equal_direct_reference_including_partial_tiles() {
        let (_, mut plants) = load_plants("offsets.txt").unwrap();
        rank_plants(&mut plants);
        let tables = make_error_tables(&plants);
        for post in [false, true] {
            let opt = Options {range: 9, y_min:62, y_max:69, ..Default::default()};
            let grid = Grid::new(&opt).unwrap();
            for tile_size in [8,16] {
                let halo = tile_halo(&plants, post, tile_size, grid.y_count).unwrap();
                for cutoff in [1,22,55,121,496] {
                    let mut expected = Vec::new();
                    for index in 0..grid.total {
                        let pos = grid.position(index);
                        let score = eval_cpu(pos, &plants, cutoff, post);
                        if score != FAIL { expected.push((pos.x,pos.y,pos.z,score)); }
                    }
                    let column_results = scan_cpu_columns(grid, 0, grid.total as u32, &plants,
                        cutoff, post, CpuLoop::Grouped, Some(&tables));
                    let mut column = column_results.into_iter().map(|(i,d)| {
                        let p = grid.position(u64::from(i)); (p.x,p.y,p.z,d)
                    }).collect::<Vec<_>>();
                    let tiles = u64::from(grid.width.div_ceil(tile_size));
                    let (_, tile_results) = scan_cpu_tile_batch::<false>(grid,tile_size,0,tiles*tiles,
                        &plants,&tables,cutoff,post,halo);
                    let (_, tile_lut_results) = scan_cpu_tile_batch::<true>(grid,tile_size,0,tiles*tiles,
                        &plants,&tables,cutoff,post,halo);
                    let mut a = tile_results.into_iter().map(|(p,d)| (p.x,p.y,p.z,d)).collect::<Vec<_>>();
                    let mut b = tile_lut_results.into_iter().map(|(p,d)| (p.x,p.y,p.z,d)).collect::<Vec<_>>();
                    expected.sort_unstable(); column.sort_unstable(); a.sort_unstable(); b.sort_unstable();
                    assert_eq!(column,expected);
                    assert_eq!(a,expected);
                    assert_eq!(b,expected);
                }
            }
        }
    }

    #[test]
    fn gpu_packed_lookup_reproduces_direct_error() {
        let (_,mut plants)=load_plants("offsets.txt").unwrap();
        rank_plants(&mut plants);
        let tables=make_error_tables(&plants);
        for (p,table) in plants.iter().zip(&tables) {
            for hash in (0..4096usize).step_by(3) {
                let x=hash as i32 & 15;
                let y=(hash as i32>>4)&15;
                let z=(hash as i32>>8)&15;
                let direct=(p.observed.x-x).abs()+(p.observed.y-y).abs()+(p.observed.z-z).abs();
                let word=u32::from_le_bytes(table[(hash/4)*4..(hash/4)*4+4].try_into().unwrap());
                let lookup=(word>>((hash%4)*8))&255;
                assert_eq!(lookup,direct as u32);
            }
        }
    }

    #[test]
    fn exact_threshold_is_exclusive() {
        let plant = Plant {relative:Pos{x:0,y:0,z:0}, observed:Offset{x:15,y:15,z:15}};
        let candidate = Pos{x:1,y:2,z:3};
        let delta = difference(candidate,plant,false);
        assert_eq!(eval_cpu(candidate,&[plant],delta,false),FAIL);
        assert_eq!(eval_cpu(candidate,&[plant],delta+1,false),delta);
    }
}
