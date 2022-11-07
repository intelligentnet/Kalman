# Kalman Filter for examples in Kalman Filter Made Easy book
Available here : https://thekalmanfilter.com/kalman-filter-made-easy-ebook/

This repository currently contains Chapter 5 and Chapter 6 code. It was created so I could consolidate my understanding of Kalman filters as I believe they
can be an important ingredient in Reinforcement Learning in continous domains.

To install Rust see http:://rustup.rs

To compile and run code:
```
cargo run --example ch5
cargo run --example ch6
```
Some next steps are :
- Make Kalman common library code
- Add additional examples
- Add acceleration / changing velocities
- Is there a common procedure that will allow parameterisation of the Kalman filter, to 'learn' the environment
- Dynamically adjust to a changing environment, and the locality in an environment (the contours of the n-dimension space, in effect). 
  Is there some signal in the noise?
- and lot's more.... Then integrate with Reinforcement Learning with continuous environments

Baby steps.

#############################################################################

There is a new program in examples. This is an extra and not part of the book.
It runs some models and filters them, they are all linear at present. They are here as examples to be built on later to better demonstrate how to build models
and parameterise Kalman filters. They help build intuition, so play. Chapter 5 and 6 code will be incorporated cleanly later. It already works with the bitcoin spreadsheet data.

```
cargo run --example examples [line:sine:parabola:bitcoin] [n=<no of iterations, default 20>] [scale=n where n is a decimal scale factor for sensor noise] [gt: show ground truth graph] [motion: show motion graph] [scatter: show Kalman as a scatter graph, and other graphs as lines]
```

Measurement data and the Kalman filtered data is always graphed.

Note: There are 4 steps in algorithmic data modelling, the ground truth data is the pure 'function', then there is motion that will see some noise (think of a car, there are bumps, small rises and falls in the road, obstructions to avoid, etc. On top of that there is measurement data, sensors are not perfect. So, there are two levels of noise. Obviously when using something like bitcoin data what is the ground truth? There is non-observable and we only have the measurement data. A scale of 0.0 is effectively running a Kalman filter on the ground truth data. 

so concretely :

```
cargo run --example examples line n=20 scale=1 gt motion scatter
```

This will do the obvious. Note then n is ignored for the fixed size bitcoin data, as that is fixed. This does raise a thought, how well does the model predict the bitcoin price 4 weeks later say? Not implemented yet and don't be too optimistic!. If only it were that easy. Perhaps that can be explored later...
