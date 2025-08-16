## INTRODUCTORY SECTION

In Minecraft, grass plants can be used to get the coordinates of a particular area.
They are particularly useful in older versions, before the introduction of texture rotations in 1.8.

Offsets are always 4 bit numbers (0..=15).
Each grass plant (also known as "tall grass" until 1.12.2, "grass" since 1.13, and "short grass" since 1.20.3), has a block position, and an offset, which gets applied to its block position, changing it slightly.
Grass plants use three-dimensional (XYZ) offsets.

Prior to 1.7.2, the only other block to use some offsets was the fern plant, which uses three-dimensional offsets, like grass.
This is due to the fact fern and grass plants are different enum types of the same block, "tallgrass" (at least in older versions).
In 1.7.2 many new flowers and double plants were added and they all started to use two-dimensional offsets.

Now, grass uses 3D offsets, one for x, y, z, but this doesn't tell us much about the way it's calculated...
It turns out the algorithm changed between versions but it's always dependent on some block coordinates. The same is true for flowers and double plants, in fact they all use the same algorithm to determine offsets.
Let's have a look at the function:
```java
public static long getCoordinateRandom(int x, int y, int z) {
	long i = (long)(x * 3129871) ^ (long)z * 116129781L ^ (long)y;
	i = i * i * 42317861L + i * 11L;
	return i;
}
```
In 1.8 the algorithm changed a little. The getCoordinateRandom function is always called with y = 0 for grass, flowers and double plants.
The result of the function gets used like this:
```python
xOffset = (i >> 16 & 15)
yOffset = (i >> 20 & 15)
zOffset = (i >> 24 & 15)
```
This means we can use the reconstructed offsets of some plants to get their world coordinates from some in-game screenshots, by simply bruteforcing the whole 3D (or 2D, if y = 0) search space within a specified x, y, z range.
In order to do this, we also need to know the relative position of these plants.

Flowers and double plants use 2D offsets so they won't use any yOffset.
So the main difference between grass (or fern) and the other plants, is that the former uses more data from the result of this function, so it's a more powerful filter.
You can use anything though, flowers, double plants, grass, fern, even texture rotations (starting from 1.8)!
In particular, texture rotations are a much weaker filter than any of the alternatives, but are by far the easiest to gather.
This simply means you'll need more texture rotations than 2D offsets, and more 2D offsets than 3D offsets, to pinpoint the precise world position featured in a screenshot.

Gathering offsets will most likely require the use of a perspective reverser AND an overlay.

I've checked the code (and in game) and it looks like double plants use y = 0 when calling getCoordinateRandom, even in 1.7.10. This means you can't get the y coordinate from them, ever.

So to recap:
In 1.7.10 and older versions:
the offsets are calculated using x, y and z (3D search space for grass), apart for double plants, for which only x and z are used.
grass and fern are the only plants using all three offsets (even the one for y), while every other plant uses just the ones for x and z.

In 1.8 and more recent versions:
the offsets are calculated using only x and z (2D search space for grass).
just like in 1.7.10, grass and fern are the only plants using all three offsets (even the one for y), while every other plant only uses the ones for x and z.

## Pre-existing tools
Another member of the Minecraft@Home community by the name of polymetric created some tools to find world coordinates based on grass.
These tools are polymetric's grassfinder and manual grass mod for Minecraft 1.16.5.

As I mentioned earlier:
we can use the reconstructed offsets of some plants to get their world coordinates from some in-game screenshots, by simply bruteforcing the whole 3D (or 2D, if y = 0) search space within a specified x, y (y=0 if 2D), z range.

And sure enough, this is what the grassfinder is about.
I'm sure it worked perfectly for polymetric but I've changed some stuff I didn't like here: TODO link my fork.

I'll talk about the main differences between the two versions.

This line:
```rust
let spiral = ChebyshevIterator::new(0, 0, 2048);
```
creates a spiralling iterator.
This is the same as iterating over a -2048..=2048 range for x and z, but checking first the positions closest to spawn, progressively turning away from it.
This is often used when you're interested in getting the closest match to a particular point (say, the spawn), and can then return early, without having to check the entire -2048..=2048 range.
In this case there's no early return though, because we don't know what the actual coordinates are, and we need to check the entire range (otherwise we would just use a smaller one), so using this type of iterator doesn't actually do much.

Anyway, usually, our ranges look more like -10000..10000, than -2048..2048, which means there are a lot more positions to check.
Compare 20000^2 \* 66 = 26 billion vs 4096^2 \* 66 = 1 billion.
The code took ~9 minutes to check the former on my hardware (I was checking these positions against 5 position-offset pairs).

What could help improving the program's performance is parallelization. Which is easier to carry on using normal iterators.
It's very easy to parallelize a Rust program using rayon, so I did just that, and got a 4x performance boost (my CPU only has 4 cores).

It still took 140 seconds though, so I improved the position checking code.

Due to the fact it's quite difficult to recreate a grass plant's offsets in a precise way, polymetric's code doesn't just check for exact matches, but rather, calculates the delta (or difference) between the offsets of the input positions and those of the other positions we are checking them against.
Let's see a practical example of this.
```json
{"input_positions": [
	{"pos": [0,0,0], "offsets": [13,0,15]}
]}
```
To make it easier, let's assume we only know the offsets of one grass plant.
This input data would be checked against a lot of different coordinates based on the input range.
Let's assume those offsets get checked against the offsets of another grass plant at coordinates 219, 73, -1200. The plant at that position could then have these offsets: [13, 1, 15].
This is how the program would calculate the delta value: abs(input_offsets[0] - test_offsets[0]) + abs(input_offsets[1] - test_offsets[1]) + abs(input_offsets[2] - test_offsets[2]), which equates to 1.
This means the two positions have an offset delta of 1. Pretty close.
Clearly, this example uses one plant only, you would normally have many of those and calculate the average delta, by summing all the offset deltas and dividing them by the total number of grass plants in the input data.
If we had 5 grass plants (input positions and offsets) each one with offset_delta = 1 (when matched against a position in the searchspace), total_offset_delta would be 5, and the average delta would still be 1.

So I replaced this code (*functionally equivalent code rewrite shown here):
```rust
fn get_pos_delta(testpos: Position, rows: &[(Position, Offset)], recorigin: Position, version: Version, grass_count: usize) -> f64 {
    let mut delta: f64 = 0.0;

    for (pos, off) in rows.iter() {
        let pos_abs = testpos + *pos - recorigin;
        let testoff = grass_offset_from_pos(pos_abs, version);

        delta += (*off - testoff).abs() as f64;
    }

    delta /= grass_count as f64;
    delta
}
```
with this code
```rust
fn get_pos_delta(testpos: Position, rows: &[(Position, Offset)], recorigin: Position, version: Version, grass_count: usize, max_total_delta: f64) -> Option<f64> {
    let mut total_delta: f64 = 0.0;
    // max_total_delta = max_avg_delta * grass_count

    for (pos, off) in rows.iter() {
        let pos_abs = testpos + *pos - recorigin;
        let testoff = grass_offset_from_pos(pos_abs, version);

        total_delta += (*off - testoff).abs() as f64;
        if (total_delta >= max_total_delta) {
            return None;
        }
    }

    let avg_delta = total_delta / grass_count as f64;
    Some(avg_delta)
}
```
which optimized away the need to calculate the grass offsets of every position (we had 26 billion of those), `grass_count` times (5 in my case, for a total of 130 billion calculations vs ~46 billion calculations).
Basically, the more positions we had, the slower the program would have been.
This led to another ~3x performance increase (the more positions you use, the slower the original code is compared to this new version).

I also made the program just take the first position in offsets.txt (the file with our input data) and use it as our origin pos so we don't have to input it ourselves as a program argument, and changed the version enum Post1_12 (the one which makes the program use x and z coords only) to Post1_8 as per previous findings.

The manual grass mod was made by polymetric for Fabric and Minecraft 1.16.5, and it's a pretty good mod. I wanted and kind of had to create my own though, so I eventually did that.
Before talking about that though, I think it would help to talk about my particular use case for these tools.

## The screenshot
I like Minecraft screenshots, and I just loved this one.
The house, the lake, the trees! The storm raging on... and Ash Ketchum... I had to find this world.
I was also kinda skeptical of whether I'd be able to find it or not: the player's using a texture pack, it's a messy compressed screenshot and not of a crazy high res to start with. I'd never really used grass offsets, and there were no clouds to help me filter z coords.
The only really usable features are trees (maybe some double plants? Possibly the clay patch?) and grass to get coords, albeit in retrospect, I think I should have also used those double plants.
Many trees are shown; the hard part is getting the coordinates of the area.

Because of the compression artifacts, the rain, and the low quality of the screenshot in general, I knew I had to use an overlay and a perspective reverser, and had to be accurate while I was at it.
I also needed to find the original texture pack, Bachale helped in this regard by stating the texture pack's name and I was able to find the precise version used by the player: Faithful64x64 by HiTeeN, v1.4.0 for Minecraft 1.7.2.
