use rayon::prelude::*;
use std::fs;
use std::error::Error;
use std::env;
use std::ops::*;

const PATH: &'static str = "offsets.txt";

fn check_pos(testpos: Position, rows: &[(Position, Offset)], recorigin: Position, version: Version) -> bool {
    for (pos, off) in rows.iter() {
        let pos_abs = testpos + *pos - recorigin;
        let testoff = grass_offset_from_pos(pos_abs, version);

        if testoff != *off {
            return false;
        }
    }
    return true;
}

fn get_pos_delta(testpos: Position, rows: &[(Position, Offset)], recorigin: Position, version: Version, grass_count: usize, max_total_delta: f64) -> Option<f64> {
    let mut total_delta: f64 = 0.0;
    let temp_pos = testpos - recorigin;

    for (pos, off) in rows.iter() {
        let pos_abs = temp_pos + *pos;
        let testoff = grass_offset_from_pos(pos_abs, version);

        total_delta += (*off - testoff).abs() as f64;
        if total_delta >= max_total_delta {
            return None;
        }
    }

    let avg_delta = total_delta / grass_count as f64;
    Some(avg_delta)
}

fn main() {
    // you probably want to edit this
    let spawnrange = 15000;
    //let yrange = (62, 128);
    // and this
    let yrange = (62, 70);

    let args: Vec<String> = env::args().collect();

    let rows = load_grass_positions().expect("wrong file format");
    let grass_count = rows.len();
    if grass_count == 0 {
        panic!("no grass found in file offsets.txt");
    }
    let recorigin = rows[0].0; // recorigin should always be the position of the first grass plant

    // setting this to false makes everything much faster (2x?)
    let delta_mode = true;
    let max_avg_delta = 5.0;
    let max_total_delta = max_avg_delta * grass_count as f64;

    let mut version = Version::PostB1_5;
    for arg in args {
        if arg == "--post-1.12" {
            version = Version::Post1_12;
        }
    }
    println!("running with version {:?}", version);

    let (y_min, y_max) = match version {
        Version::PostB1_5 => yrange,
        Version::Post1_12 => (0, 1),
    };

    let y_range: Vec<_> = (y_min..y_max).collect();
    let xz_range: Vec<_> = (-spawnrange..spawnrange).collect();

    xz_range.par_iter().for_each(|&x| {
        xz_range.par_iter().for_each(|&z| {
            y_range.iter().for_each(|&y| {
                let testpos = Position { x, y, z };

                if delta_mode {
                    let maybe_avg_delta = get_pos_delta(testpos, &rows, recorigin, version, grass_count, max_total_delta);
                    match maybe_avg_delta {
                        Some(avg_delta) => {
                            println!(
                                "{:>8}{:>8}{:>8} has an average grass delta of {:.3}, delta between positions is {:>8}{:>8}{:>8}",
                                x, y, z, avg_delta, x-recorigin.x, y-recorigin.y, z-recorigin.z
                            );
                        },
                        _ => (),
                    }
                } else if check_pos(testpos, &rows, recorigin, version) {
                    println!(
                        "{:>8}{:>8}{:>8} matches, delta between positions is {:>8}{:>8}{:>8}",
                        x, y, z, x-recorigin.x, y-recorigin.y, z-recorigin.z
                    );
                }
            });
        });
    });
}

const X_MULT: i32 = 0x2fc20f;
const Z_MULT: i32 = 0x6ebfff5;
const LCG_MULT: i64 = 0x285b825;
const LCG_ADDEND: i64 = 11;

fn grass_offset_from_pos(p: Position, version: Version) -> Offset {
    grass_offset(p.x, p.y, p.z, version)
}

fn grass_offset(x: i32, y: i32, z: i32, version: Version) -> Offset {
    let seed = match version {
        Version::PostB1_5 => get_coord_random(x, y, z),
        Version::Post1_12 => get_coord_random(x, 0, z),
    };

    Offset {
        x: (seed >> 16 & 15) as i8,
        y: (seed >> 20 & 15) as i8,
        z: (seed >> 24 & 15) as i8,
    }
}

fn get_coord_random(x: i32, y: i32, z: i32) -> i64 {
    let mut seed = (x * X_MULT) as i64 ^ (z * Z_MULT) as i64 ^ y as i64;
    seed = seed * seed * LCG_MULT + seed * LCG_ADDEND;
    return seed;
}

// // returns a tuple of the input integer offsets converted to
// // actual position offsets (1.0 is equal to 1 block)
// fn off_itof_xyz(x: i8, y: i8, z: i8) -> (f64, f64, f64) {
// 	(
// 		map(x as f64, 0.0, 15.0, -0.25, 0.25),
// 		map(y as f64, 0.0, 15.0, -0.20, 0.00),
// 		map(z as f64, 0.0, 15.0, -0.25, 0.25),
// 	)
// }

// fn off_ftoi_xyz(x: f64, y: f64, z: f64) -> (i8, i8, i8) {
// 	(
// 		map(x, -0.25, 0.25, 0.0, 15.0) as i8,
// 		map(y, -0.2, 0.0, 0.0, 15.0) as i8,
// 		map(z, -0.25, 0.25, 0.0, 15.0) as i8,
// 	)
// }

// // standard linear interp
// fn map(
// 	x: f64,
// 	in_min: f64,
// 	in_max: f64,
// 	out_min: f64,
// 	out_max: f64
// ) -> f64 {
// 	(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
// }

// Struct and Enum definitions

#[derive(Copy, Clone, Debug)]
struct Position {
    x: i32,
    y: i32,
    z: i32,
}

#[derive(Copy, Clone, Debug)]
struct Offset {
    x: i8,
    y: i8,
    z: i8,
}

impl Offset {
    fn abs(&self) -> i8 {
        self.x.abs() + self.y.abs() + self.z.abs()
    }
}

impl Sub for Offset {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl PartialEq for Offset {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x &&
        self.y == other.y &&
        self.z == other.z
    }
}

impl Sub for Position {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Add for Position {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

fn load_grass_positions() -> Result<Vec<(Position, Offset)>, Box<dyn Error>> {
    let contents = fs::read_to_string(PATH)?;

    Ok(contents
        .lines()
        .map(|line| {
            let mut line = line
                .split_whitespace();

            (
                Position {
                    x: line.next().unwrap().parse::<i32>().unwrap(),
                    y: line.next().unwrap().parse::<i32>().unwrap(),
                    z: line.next().unwrap().parse::<i32>().unwrap(),
                },
                Offset {
                    x: line.next().unwrap().parse::<i8>().unwrap(),
                    y: line.next().unwrap().parse::<i8>().unwrap(),
                    z: line.next().unwrap().parse::<i8>().unwrap(),
                }
            )
        })
        .collect::<Vec<(Position, Offset)>>())
}

#[derive(Clone, Copy, Debug)]
enum Version {
    PostB1_5,
    Post1_12,
}
