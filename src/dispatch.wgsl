// Converts stage-one survivor count into GPU indirect dispatch dimensions.
struct Counter { count: atomic<u32> };
struct Dispatch { x: u32, y: u32, z: u32 };
@group(0) @binding(0) var<storage, read_write> survivors: Counter;
@group(0) @binding(1) var<storage, read_write> dispatch: Dispatch;
@compute @workgroup_size(1)
fn main() {
    let n = atomicLoad(&survivors.count);
    dispatch.x = (n + 63u) / 64u;
    dispatch.y = 1u;
    dispatch.z = 1u;
}
