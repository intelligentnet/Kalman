use ndarray::prelude::*;
//use ndarray_linalg::solve::Inverse;
use rand::prelude::*;
use num_traits::{Float, Zero};

trait Inverse<T: Float> {
    fn det(&self) -> T;
    fn inverse(&self) -> Option<Self> where Self: Sized;
}

impl<T: Float + std::fmt::Debug> Inverse<T> for Array2<T> {
    fn det(&self) -> T {
        match self.shape() {
            [2, 2] => {
                self[(0, 0)] * self[(1, 1)] - self[(1, 0)] * self[(0, 1)]
            }
            [3, 3] => {
                self[(0, 0)] * (self[(1, 1)] * self[(2, 2)] - self[(2, 1)] * self[(1, 2)]) -
                self[(0, 1)] * (self[(1, 0)] * self[(2, 2)] - self[(1, 2)] * self[(2, 0)]) +
                self[(0, 2)] * (self[(1, 0)] * self[(2, 1)] - self[(1, 1)] * self[(2, 0)])
            }
            _ => Zero::zero()
        }
    }

    fn inverse(&self) -> Option<Self> {
        let det = self.det();
        if det != Zero::zero() {
            let recip = det.recip();

            match self.shape() {
                [2, 2] => {
                    Some(array![
                        [self[(1, 1)] * recip, -self[(0, 1)] * recip],
                        [-self[(1, 0)] * recip, self[(0, 0)] * recip],
                    ])
                }
                [3, 3] => {
                    let mut res = Array2::<T>::zeros((3, 3));
                    res[(0, 0)] = (self[(1, 1)] * self[(2, 2)] - self[(2, 1)] * self[(1, 2)]) * recip;
                    res[(0, 1)] = (self[(0, 2)] * self[(2, 1)] - self[(0, 1)] * self[(2, 2)]) * recip;
                    res[(0, 2)] = (self[(0, 1)] * self[(1, 2)] - self[(0, 2)] * self[(1, 1)]) * recip;
                    res[(1, 0)] = (self[(1, 2)] * self[(2, 0)] - self[(1, 0)] * self[(2, 2)]) * recip;
                    res[(1, 1)] = (self[(0, 0)] * self[(2, 2)] - self[(0, 2)] * self[(2, 0)]) * recip;
                    res[(1, 2)] = (self[(1, 0)] * self[(0, 2)] - self[(0, 0)] * self[(1, 2)]) * recip;
                    res[(2, 0)] = (self[(1, 0)] * self[(2, 1)] - self[(2, 0)] * self[(1, 1)]) * recip;
                    res[(2, 1)] = (self[(2, 0)] * self[(0, 1)] - self[(0, 0)] * self[(2, 1)]) * recip;
                    res[(2, 2)] = (self[(0, 0)] * self[(1, 1)] - self[(1, 0)] * self[(0, 1)]) * recip;
                    Some(res)
                }
                _ => None
            }
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct Measurement {
    x: f64,
    y: f64,
    cov: Array2<f64>,
    t_r: f64,
    t_b: f64,
    r: f64,
    b: f64,
    z: Array2<f64>,
}

impl Measurement {
    // Bearing standard deviation = 9 milliradians (in degrees)
    const SIG_B: f64 = 0.009 * 180.0 / std::f64::consts::PI;
    const SIG_R: f64 = 30.0; // Range Standard Deviation = 30 meters
    const SIG_B_Q: f64 = 0.25 * Measurement::SIG_B;
    const SIG_R_Q: f64 = 0.25 * Measurement::SIG_R;
    const DT: f64 = 1.0;
    const PI2: f64 = 180.0 / std::f64::consts::PI;
    const X_VEL: f64 = 22.0;
    const Y_VEL: f64 = 0.0;

    fn new() -> Measurement {
        Measurement {
            x: 2900.0,
            y: 2900.0,
            cov: array![[0.0, 0.0], [0.0, 0.0]],
            t_r: 0.0,
            t_b: 0.0,
            r: 0.0,
            b: 0.0,
            z: array![[0.0], [0.0]],
        }
    }

    fn get(&mut self) {
        let mut rng = rand::thread_rng();
        // Compute the new x and y position data with the assumption all of the
        // velocity motion is in the x direction i.e. linear motion
        self.x += Measurement::DT * Measurement::X_VEL;
        self.y += Measurement::DT * Measurement::Y_VEL;
        // Compute the sum of the squares of x and y as an intermediate step
        // before computing the range and storing it off
        self.t_r = (self.x * self.x + self.y * self.y).sqrt();
        // Compute the azimuth (or bearing) with the arctan2 function and convert
        // it to degrees. Then store this azimuth data
        self.t_b = self.x.atan2(self.y) * Measurement::PI2;

        // Compute the error for each measurement
        // By taking the max between .25 of the defined standard deviation and
        // the randomly generated normal error, it guarantees an error
        let temp_sig_b = (Measurement::SIG_B * rng.gen::<f64>()).max(Measurement::SIG_B_Q);
        let temp_sig_r = (Measurement::SIG_R * rng.gen::<f64>()).max(Measurement::SIG_R_Q);
        // for debugging
        //let temp_sig_b = (Measurement::SIG_B * 0.5).max(Measurement::SIG_B_Q);
        //let temp_sig_r = (Measurement::SIG_R * 0.5).max(Measurement::SIG_R_Q);

        // Save off the measurement values for bearing and range as a Function
        // of the true value + the error generated above
        self.r = self.t_r + temp_sig_r;
        self.b = self.t_b + temp_sig_b;

        self.z[[0,0]] = self.r;
        self.z[[1,0]] = self.b;

        self.cov[[0,0]] = temp_sig_r * temp_sig_r;
        self.cov[[1,1]] = temp_sig_b * temp_sig_b;
    }
}

trait FilterNew {
    fn new() -> Filter;
}

impl FilterNew for Filter {
    fn new() -> Filter {
        Filter {
            x: Array2::<f64>::zeros((4, 1)),
            p: Array2::<f64>::zeros((4, 4)),
            a: Array2::<f64>::eye(4),
            b: Array2::<f64>::zeros((4, 1)),
            h: Array2::<f64>::zeros((2, 4)),
            r: Array2::<f64>::zeros((2, 2)),
            q: Array2::<f64>::zeros((4, 4)),
        }
    }
}

/*
trait FilterLinearFunctions {
    fn f(&mut self, m: &mut Measurement) -> Array2<f64>;
    fn h(&mut self, m: &mut Measurement) -> Array2<f64>;
}

impl FilterLinearFunctions for Filter {
    fn f(&mut self) -> Array2<f64> {
        self.a.dot(&self.x)
    }

    fn h(&mut self, x_prime: &Array2<f64>) -> Array2<f64> {
        self.h.dot(x_prime)
    }
}
*/

trait FilterFunctions {
    fn f(&mut self, m: &mut Measurement) -> Array2<f64>;
    fn h(&mut self, m: &mut Measurement, x_prime: &Array2<f64>) -> Array2<f64>;
}

impl FilterFunctions for Filter {
    fn f(&mut self, m: &mut Measurement) -> Array2<f64> {
        // Form state to measurement transition matrix
        let x_prime = self.a.dot(&self.x);
        let x = x_prime[[0, 0]];
        let y = x_prime[[1, 0]];
        let den = x * x + y * y;
        let densq = den.sqrt();
        self.h = array![
            [x / densq, y / densq, 0.0, 0.0],
            [y / den, -x / den, 0.0, 0.0]
        ];
        // Measurement covariance matrix
        self.r = m.cov.to_owned();

        x_prime
    }

    fn h(&mut self, _m: &mut Measurement, x_prime: &Array2<f64>) -> Array2<f64> {
        // Convert the predicted cartesian state to polar range and azimuth
        let x = x_prime[[0, 0]];
        let y = x_prime[[1, 0]];
        let r = (x * x + y * y).sqrt();
        let b = x.atan2(y) * 180.0 / std::f64::consts::PI;

        array![[r], [b]]
    }
}

trait FilterInit {
    fn first(&mut self, m: &mut Measurement) -> Array2<f64>;
    fn second(&mut self, m: &mut Measurement) -> Array2<f64>;
}

impl FilterInit for Filter {
    fn first(&mut self, m: &mut Measurement) -> Array2<f64> {
        // compute position values from measurements
        // x = r*sin(b)
        let temp_x = m.r * (m.b * std::f64::consts::PI / 180.0).sin();
        // y = r*cos(b)
        let temp_y = m.r * (m.b * std::f64::consts::PI / 180.0).cos();
        // State vector - initialize position values
        self.x = array![[temp_x], [temp_y], [0.0], [0.0]];
        // State transistion matrix - linear extrapolation assuming constant velocity
        self.a[[0, 2]] = Measurement::DT;
        self.a[[1, 3]] = Measurement::DT;
        // Measurement covariance matrix
        self.r = m.cov.to_owned();

        return Array2::zeros((4, 2));
    }

    fn second(&mut self, m: &mut Measurement) -> Array2<f64> {
        // compute position values from measurements
        // x = r*sin(b)
        let temp_x = m.r * (m.b * std::f64::consts::PI / 180.0).sin();
        // y = r*cos(b)
        let temp_y = m.r * (m.b * std::f64::consts::PI / 180.0).cos();
        // State vector - initialize position values
        self.x = array![
            [temp_x],
            [temp_y],
            [(temp_x - self.x[[0, 0]]) / Measurement::DT],
            [(temp_y - self.x[[1, 0]]) / Measurement::DT]
        ];
        // State covariance matrix - initialized to zero for first update
        self.p[[0, 0]] = 100.0;
        self.p[[1, 1]] = 100.0;
        self.p[[2, 2]] = 250.0;
        self.p[[3, 3]] = 250.0;
        // Measurement covariance matrix
        self.r = m.cov.to_owned();
        // System error matrix - initialized to zero matrix for first update
        self.q[[0, 0]] = 20.0;
        self.q[[1, 1]] = 20.0;
        self.q[[2, 2]] = 4.0;
        self.q[[3, 3]] = 4.0;

        return Array2::zeros((4, 2));
    }
}

#[derive(Debug)]
struct Filter {
    x: Array2<f64>,
    p: Array2<f64>,
    // z is a Measurement so input only
    a: Array2<f64>,
    b: Array2<f64>,
    h: Array2<f64>,
    r: Array2<f64>,
    q: Array2<f64>,
    // k is internal
}
 
impl Filter {
    fn filter(&mut self, m: &mut Measurement) -> Array2<f64> {
        // Predict mean covariance forward
        let x_prime = self.f(m) + &self.b;
        let p_prime = self.a.dot(&self.p).dot(&self.a.t()) + &self.q;

        // Compute Kalman Gain
        let s = self.h.dot(&p_prime).dot(&self.h.t()) + &self.r;
        //let k = p_prime.dot(&self.h.t()).dot(&s.inv().unwrap());
        let k = p_prime.dot(&self.h.t()).dot(&s.inverse().unwrap());
        // Estimate new State
        let h = self.h(m, &x_prime);
        self.x = &x_prime + &k.dot(&(&m.z - &h));
        // Estimate new Covariance
        self.p = &p_prime - k.dot(&self.h).dot(&p_prime);

        k
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct Measure {
    t_r: f64,
    t_b: f64,
    r: f64,
    b: f64,
    x: f64,
    y: f64,
    est_pos: Array1<f64>,
    est_vel: Array1<f64>,
    pos_sigma: Array1<f64>,
    pos_gain: Array1<f64>,
    vel_gain: Array1<f64>,
}

impl Measure {
    fn new(m: &mut Measurement, filt: &mut Filter, step: u32) -> Measure {
        m.get();
        let k = match step {
            0 => filt.first(m),
            1 => filt.second(m),
            _ => filt.filter(m),
        };

        let est_pos = array![filt.x[[0, 0]], filt.x[[1, 0]]];
        let est_vel = array![filt.x[[2, 0]], filt.x[[3, 0]]];
        let pos_sigma = array![filt.p[[0, 0]].sqrt(), filt.p[[1, 1]].sqrt()];
        let pos_gain = array![k[[0, 0]], k[[1, 0]]];
        let vel_gain = array![k[[2, 0]], k[[3, 0]]];

        Measure {
            t_r: m.t_r,
            t_b: m.t_b,
            r: m.r,
            b: m.b,
            x: m.x,
            y: m.y,
            est_pos,
            est_vel,
            pos_sigma,
            pos_gain,
            vel_gain,
        }
    }
}

#[allow(unused_variables)]
fn main() {
    let mut measurement = Measurement::new();
    let mut filter = Filter::new();
    let m: Vec<Measure> = (0..25)
        .map(|k| Measure::new(&mut measurement, &mut filter, k))
        .collect();
    // get loads of data out for plotting
    let ex: Vec<f64> = m.iter().map(|i| i.est_pos[0] - i.x).collect();
    let ey: Vec<f64> = m.iter().map(|i| i.est_pos[1] - i.y).collect();
    let x_pos: Vec<f64> = m.iter().map(|i| i.est_pos[0]).collect();
    let y_pos: Vec<f64> = m.iter().map(|i| i.est_pos[1]).collect();
    let x_vel: Vec<f64> = m.iter().map(|i| i.est_vel[0]).collect();
    let y_vel: Vec<f64> = m.iter().map(|i| i.est_vel[1]).collect();
    let actual_r: Vec<f64> = m.iter().map(|i| i.t_r).collect();
    let actual_b: Vec<f64> = m.iter().map(|i| i.t_b).collect();
    let est_r: Vec<f64> = m.iter().map(|i| i.r).collect();
    let est_b: Vec<f64> = m.iter().map(|i| i.b).collect();
    let x_pos_gain: Vec<f64> = m.iter().map(|i| i.pos_gain[0]).collect();
    let y_pos_gain: Vec<f64> = m.iter().map(|i| i.pos_gain[1]).collect();
    let x_vel_gain: Vec<f64> = m.iter().map(|i| i.vel_gain[0]).collect();
    let y_vel_gain: Vec<f64> = m.iter().map(|i| i.vel_gain[1]).collect();
    let e_x_3sig: Vec<f64> = m.iter().map(|i| 3.0 * i.pos_sigma[0]).collect();
    let e_y_3sig: Vec<f64> = m.iter().map(|i| 3.0 * i.pos_sigma[1]).collect();
    let ne_x_3sig: Vec<f64> = m.iter().map(|i| 3.0 * -i.pos_sigma[0]).collect();
    let ne_y_3sig: Vec<f64> = m.iter().map(|i| 3.0 * -i.pos_sigma[1]).collect();

    simple_plot::plot!("Actual Range vs Measured Range", actual_r, est_r);
    simple_plot::plot!("Actual Azimuth vs Measured Azimuth", actual_b, est_b);
    simple_plot::plot!("Velocity Estimate On Each Measurement Update", x_vel, y_vel);
    simple_plot::plot!("X Position Estimate Error Containment", ex, e_x_3sig, ne_x_3sig);
    simple_plot::plot!("Y Position Estimate Error Containment", ey, e_y_3sig, ne_y_3sig);
    // Other useful debugging and understanding plots
    //simple_plot::plot!("Cartesian position data", x_pos, y_pos);
    //simple_plot::plot!("Position Error 3 Sigma", e_x_3sig, e_y_3sig);
    //simple_plot::plot!("Position", ex, ey);
    //simple_plot::plot!("X Kalman Gains", x_pos_gain, x_vel_gain);
    //simple_plot::plot!("Y Kalman Gains", y_pos_gain, y_vel_gain);
    //simple_plot::plot!("Gains", x_pos_gain, x_vel_gain, y_pos_gain, y_vel_gain);
}
