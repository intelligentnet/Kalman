use ndarray::prelude::*;
use ndarray_inverse::Inverse;
use rand::prelude::*;
use rand_distr::Normal;
use num_traits::Float;
extern crate plotly;
use plotly::common::Mode;
use plotly::{Plot, Scatter};
use std::env;

fn plotter(plot: &mut Plot, label: &str, states: &Vec<Array1<f64>>, scatter: bool) {
    let xa: Vec<f64> = states.iter().map(|i| i[0]).collect();
    let ya: Vec<f64> = states.iter().map(|i| i[1]).collect();
    let trace = Scatter::new(xa.to_vec(), ya.to_vec())
        .name(label)
        .mode(if scatter { Mode::Markers } else { Mode::Lines });
    plot.add_trace(trace);
}

fn linspace<T: Float + std::convert::From<i16> + std::fmt::Debug>(l: T, h: T, n: usize) -> Vec<T> {
    let size: T = (n as i16 - 1).try_into()
        .expect("too many elemnets: max is 2^15");
    let dx = (h - l) / size;

    let low = l.to_i16().unwrap();
    let high = size.to_i16().unwrap() + low;
    (low ..= high).map(|i| { dx * (i as i16).try_into().unwrap() }).collect()
}

// Generate motion data from ground truth by adding noise
fn gen_noisy_states(gt: &Vec<Array1<f64>>, stdev: &Array2<f64>) -> Vec<Array1<f64>> {
    // Must be diagonal matrix! i,e xl == yl
    let s = stdev.raw_dim();
    let l = s[0];
    let mut rng = rand::thread_rng();

    gt.into_iter().map(|i| {
        let mut res = Array1::<f64>::zeros(l);

        (0 .. l).zip(0 .. l).for_each(|(x, y)| {
            let normal = Normal::new(i[x], stdev[(x, y)]).unwrap();

            res[x] = normal.sample(&mut rng);
        });

        res
    }).collect()
}

fn predict(ma: &Array2<f64>, mb: &Array2<f64>, mq: &Array2<f64>, u_t: &Array1<f64>, mu_t: &Array1<f64>, sigma_t: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let predicted_mu = ma.dot(mu_t) + mb.dot(u_t);
    let predicted_sigma = ma.dot(&sigma_t.dot(&ma.t())) + mq;

    (predicted_mu, predicted_sigma)
}

fn kalman_gain(mh: &Array2<f64>, mr: &Array2<f64>, predicted_sigma: &Array2<f64>) -> Array2<f64> {
    let residual_covariance = mh.dot(&predicted_sigma.dot(&mh.t())) + mr;
    
    predicted_sigma.dot(&mh.t().dot(&residual_covariance.inv().unwrap()))
}

fn update(gain: &Array2<f64>, z: &Array1<f64>, mh: &Array2<f64>, predicted_mu: &Array1<f64>, predicted_sigma: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let residual_mean = z - mh.dot(predicted_mu);
    let updated_mu = predicted_mu + gain.dot(&residual_mean);
    let updated_sigma = predicted_sigma - gain.dot(&mh.dot(predicted_sigma));

    (updated_mu, updated_sigma)
}

fn filter(offset: &Array1<f64>, mz: &Vec<Array1<f64>>, mx: &mut Array1<f64>, mp: &mut Array2<f64>, ma: &Array2<f64>, mb: &Array2<f64>, mh: &Array2<f64>, mr: &Array2<f64>, mq: &Array2<f64>) -> Vec<Array1<f64>> {
        mz.iter()
        .map(|z| {
            let (mu, sigma) = predict(&ma, &mb, &mq, &offset, &mx, &mp);
            let gain = kalman_gain(&mh, &mr, &sigma);
            (*mx, *mp) = update(&gain, &z, &mh, &mu, &sigma);

            mx.clone()
        })
        .collect()
}

fn model_select(model: &str, n: usize) -> Vec<Array1<f64>> {
    fn combine(x: Vec<f64>, y: Vec<f64>) -> Vec<Array1<f64>> {
        x.clone()
            .into_iter()
            .zip(&y)
            .map(|(a,&b)| array![a, b])
            .collect()
    }

    match model {
        "line" => {
            // Straight Line
            let ground_truth_x = linspace(0.0, n as f64, n + 1);
            let ground_truth_y = ground_truth_x.clone();
            
            combine(ground_truth_x, ground_truth_y)
        },
        "sine" => {
            // Sine wave
            let no_waves = 2.0;
            let ground_truth_x = linspace(0.0, n as f64, n + 1);
            let ground_truth_y =
                ground_truth_x
                    .iter()
                    .map(|x| (2.0 * std::f64::consts::PI * x * no_waves / n as f64).sin())
                    .collect();

            combine(ground_truth_x, ground_truth_y)
        },
        "parabola" => {
            let ground_truth_x = linspace(-(n as f64 / 2.0), n as f64 - (n as f64 / 2.0), n + 1);
            let a = 0.1;
            let ground_truth_y: Vec<f64> = ground_truth_x.iter().map(|x| a * x * x).collect();

            combine(ground_truth_x, ground_truth_y)
        },
        "bitcoin" => {
            // Weekly Bitcoin prices for 2021 in USD
            bitcoin()
        },
        &_ => todo!(),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut num_steps = 20;
    let mut scale: f64 = 1.0;
    let mut model = "line";
    let mut with_gt = false;
    let mut with_motion = false;
    let mut scatter = false;
    // Unpack args
    args.iter().for_each(|s| match s.as_str() {
        "line" | "sine" | "parabola" | "bitcoin" => model = s,
        "gt" => with_gt = true,
        "motion" => with_motion = true,
        "scatter" => scatter = true,
        s => if s.starts_with("n=") {
            num_steps = s[2 ..].parse().unwrap()
        } else if s.starts_with("scale=") {
            scale = s[6 ..].parse().unwrap();
            if scale < 0.000000001 { scale = 0.000000001 };
        }
        });
    println!("Model {} for {} steps, scaled by {}", model, num_steps, scale);

    let ground_truth_states = model_select(model, num_steps);

    // Initialse motion Process Noise
    let x_motion_stdev = 0.1 * scale;
    let y_motion_stdev = 0.1 * scale;
    let mq = array![[x_motion_stdev, 0.0], [0.0, y_motion_stdev]];

    let motion_states: Vec<Array1<f64>> = if model == "bitcoin" {
        ground_truth_states.clone()
    } else {
        // Derive motion given the uncertainties of the world
        gen_noisy_states(&ground_truth_states, &mq)
    };

    // Initialse Measurement noise
    let x_measure_stddev = 0.25 * scale;
    let y_measure_stddev = 0.25 * scale;
    let mr = array![[x_measure_stddev, 0.0], [0.0, y_measure_stddev]];

    let measured_states: Vec<Array1<f64>> = if model == "bitcoin" {
        ground_truth_states.clone()
    } else {
        // Derive measurements from the motion states
        gen_noisy_states(&motion_states, &mr)
    };

    // Parameterise Kalman model for this problem
    // Note: mq and mr initialised above for data generation!
    let ma = Array2::<f64>::eye(2); // Scale each data item
    let mb = Array2::<f64>::eye(2); // Offet scaling for each data item
    let mh = Array2::<f64>::eye(2); // State to measurement scale
    let initial_rate_of_change_x = 0.1;
    let initial_rate_of_change_y = 0.1;
    let mut sigma_current = array![[initial_rate_of_change_x, 0.0],
                                   [0.0, initial_rate_of_change_y]];
    // Offsets - if any
    let offset = array![0.0, 0.0];

    // Initialise Average and covariance
    let mut mu_current = array![measured_states[0][0],
                                measured_states[0][1]];

    // Now run Kalman Filter for each time step on measured states
    let filtered_states: Vec<Array1<f64>> =
       filter(&offset, &measured_states, &mut mu_current, &mut sigma_current, &ma, &mb, &mh, &mr, &mq);

    // Plot results
    let mut plot = Plot::new();

    if with_gt {
        plotter(&mut plot, "Ground Truth", &ground_truth_states, !scatter);
    }
    if with_motion {
        plotter(&mut plot, "Motion Values", &motion_states, !scatter);
    }
    plotter(&mut plot, "Measured Values", &measured_states, !scatter);
    plotter(&mut plot, "Filtered Values", &filtered_states, scatter);

    plot.show();
 
    // Performance timer
    /*
    let mut _filtered_states: Vec<Array1<f64>> = filtered_states;
    let start = std::time::Instant::now();

    for _ in 0 .. 100000 {
        _filtered_states = filter(&offset, &measured_states, &mut mu_current, &mut sigma_current, &ma, &mb, &mh, &mr, &mq);
    }

    println!("Time taken: {:?}", start.elapsed());
    */
}

fn bitcoin() -> Vec<Array1<f64>> {
    vec!(array![0.0, 33922.96],
  	array![ 1.0, 36069.80 ],
  	array![ 2.0, 32569.85 ],
  	array![ 3.0, 35510.29 ],
  	array![ 4.0, 46481.11 ],
  	array![ 5.0, 49199.87 ],
  	array![ 6.0, 48824.43 ],
  	array![ 7.0, 48378.99 ],
  	array![ 8.0, 54824.12 ],
  	array![ 9.0, 56804.90 ],
  	array![ 10.0, 54738.95 ],
  	array![ 11.0, 58917.69 ],
  	array![ 12.0, 58192.36 ],
  	array![ 13.0, 63503.46 ],
  	array![ 14.0, 56473.03 ],
  	array![ 15.0, 55033.12 ],
  	array![ 16.0, 53333.54 ],
  	array![ 17.0, 56704.57 ],
  	array![ 18.0, 42909.40 ],
  	array![ 19.0, 38402.22 ],
  	array![ 20.0, 36684.93 ],
  	array![ 21.0, 33472.63 ],
  	array![ 22.0, 40406.27 ],
  	array![ 23.0, 32505.66 ],
  	array![ 24.0, 35867.78 ],
  	array![ 25.0, 34235.20 ],
  	array![ 26.0, 32702.03 ],
  	array![ 27.0, 29807.35 ],
  	array![ 28.0, 39406.94 ],
  	array![ 29.0, 38152.98 ],
  	array![ 30.0, 45585.03 ],
  	array![ 31.0, 44695.36 ],
  	array![ 32.0, 47706.12 ],
  	array![ 33.0, 47166.69 ],
  	array![ 34.0, 46811.13 ],
  	array![ 35.0, 47092.49 ],
  	array![ 36.0, 40693.68 ],
  	array![ 37.0, 41034.54 ],
  	array![ 38.0, 51514.81 ],
  	array![ 39.0, 56041.06 ],
  	array![ 40.0, 64261.99 ],
  	array![ 41.0, 60363.79 ],
  	array![ 42.0, 63226.40 ],
  	array![ 43.0, 66971.83 ],
  	array![ 44.0, 60161.25 ],
  	array![ 45.0, 57569.07 ],
  	array![ 46.0, 57005.43 ],
  	array![ 47.0, 50700.09 ],
  	array![ 48.0, 46612.63 ],
  	array![ 49.0, 48936.61 ],
  	array![ 50.0, 47588.86 ],
  	array![ 51.0, 45897.57 ])
}
