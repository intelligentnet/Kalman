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

There is a new program in examples. This is an extra and not part of the book.
It runs some models and filters them, they are all linear at present. They are here as examples to be built on later to better demonstrate how to build models
and parameterise Kalman filters. Chapter 5 and 6 code will be incorporated 
cleanly later. It already works with the bitcoin spreadsheet data.

```
cargo run --example examples [line:sine:parabola:bitcoin] [n=<no of iterations, default 10>]
```

so concretely :

```
cargo run --example examples line n=10
```

This will do the obvious. Obviously n is ignored for the fixed size bitcoin data. This does raise a thought, how well does the model predict the bitcoin price 4 weeks later say? Not implemented yet and don't be too optimistic!. If only it were that easy.
