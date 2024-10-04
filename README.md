# grassfinder
finds coords of a minecraft screenshot using the offsets of tall grass. use [manualgrass](https://github.com/polymetric/mcah-manualgrass) to acquire these offsets


# Differences Between This Fork and the Original Program (Polymetric's Version) by ChatGPT

This fork introduces several key changes and optimizations compared to the original program (Polymetric's version). Here is a detailed breakdown of the improvements:

1. **Early Exit Optimization**

   - The `get_pos_delta` function in this fork introduces an early exit mechanism if `total_delta` exceeds `max_total_delta`. This allows the computation to terminate prematurely when an unfavorable result is evident, thereby significantly reducing unnecessary iterations and saving valuable computation time. In the original program, each offset difference was calculated for every single grass plant, for every single block in the 3D range in the list before comparing the total sum to the maximum value, which made the total execution time grow exponentially based on the number of grass blocks in the formation. The lack of an early exit mechanism meant that the more grass blocks there were, the exponentially longer the program would take to execute leading to a computationally infeasible number of iterations.

2. **Parallel Computation**

   - This fork leverages the `rayon` crate for parallel processing (`use rayon::prelude::*`), whereas the original program lacks any parallel computation. By using parallel processing, the fork is able to efficiently handle larger datasets and speed up computations considerably, especially for operations involving heavy iteration.

3. **Modular Code Design**

   - This fork introduces two new helper functions: `check_pos` and `get_pos_delta`:
     - **`check_pos`**: Determines if a given test position matches the expected offsets for all rows.
     - **`get_pos_delta`**: Computes the delta between the expected and actual offsets, with an optimization that exits early if the delta exceeds a specified threshold (`max_total_delta`).
   - These helper functions significantly improve the modularity of the code. By breaking down key computations into reusable components, the fork reduces code duplication and improves maintainability.

4. **Parameter Adjustability**

   - In this fork, key parameters such as `spawnrange` and `yrange` are explicitly defined and marked as adjustable (`let spawnrange = 15000; let yrange = (62, 70);`). This enhances flexibility, making it easier for developers to test different configurations.&#x20;

5. **Optimized Iteration**

   - The original program used a `ChebyshevIterator` (`ChebyshevIterator::new(0, 0, 2048)`) to iterate through possible test positions orderly from the spawn. In this fork, the `ChebyshevIterator` was replaced by a normal iterator to make it more easily parallelizable with `rayon`. This results in more efficient processing of positions, especially when combined with the other optimizations.

6. **Improved Readability and Maintainability**

   - By adding modular functions (`check_pos` and `get_pos_delta`) and reducing code repetition, this fork makes the code more readable and easier to maintain. These improvements lower the chances of errors, simplify future modifications, and make the program structure  clearer.

## Summary

This fork builds on the incredible work done by Polymetric. The original program laid a solid foundation with great vision, and this fork aims to further those ideas by making specific performance improvements. By incorporating parallel processing, modular functions, adjustable parameters, and an optimized early exit mechanism, this fork strives to enhance efficiency and maintainability while staying true to the original vision. These optimizations are meant to honor the innovative work of Polymetric's work, making it even more suitable for complex, large-scale scenarios.
