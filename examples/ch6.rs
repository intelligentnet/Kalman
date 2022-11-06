use ndarray::prelude::*;
use ndarray_inverse::Inverse;
use rand::prelude::*;
extern crate plotly;
use plotly::common::Mode;
use plotly::{Plot, Scatter};

#[derive(Debug)]
struct Measurement {
    x: f64,
    y: f64,
    cov: Array2<f64>,
    t_r: f64,
    t_b: f64,
    r: f64,
    b: f64,
    mz: Array2<f64>,
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

    fn new() -> Self {
        Measurement {
            x: 2900.0,
            y: 2900.0,
            cov: array![[0.0, 0.0], [0.0, 0.0]],
            t_r: 0.0,
            t_b: 0.0,
            r: 0.0,
            b: 0.0,
            mz: array![[0.0], [0.0]],
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
        // Compute the azimuth (or bearing) with the arctan2 function and
        // convert it to degrees. Then store this azimuth data
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

        self.mz[[0, 0]] = self.r;
        self.mz[[1, 0]] = self.b;

        self.cov[[0, 0]] = temp_sig_r * temp_sig_r;
        self.cov[[1, 1]] = temp_sig_b * temp_sig_b;
    }
}

#[derive(Debug)]
struct Filter {
    mx: Array2<f64>,
    mp: Array2<f64>,
    // mz is a Measurement so input only
    ma: Array2<f64>,
    mb: Array2<f64>,
    mh: Array2<f64>,
    mr: Array2<f64>,
    mq: Array2<f64>,
    // mk is internal
}

trait FilterNew {
    fn new() -> Self;
}

impl FilterNew for Filter {
    fn new() -> Self {
        Filter {
            mx: Array2::<f64>::zeros((4, 1)),
            mp: Array2::<f64>::zeros((4, 4)),
            ma: Array2::<f64>::eye(4),
            mb: Array2::<f64>::zeros((4, 1)),
            mh: Array2::<f64>::zeros((2, 4)),
            mr: Array2::<f64>::zeros((2, 2)),
            mq: Array2::<f64>::zeros((4, 4)),
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
        self.ma.dot(&self.mx)
    }

    fn h(&mut self, x_prime: &Array2<f64>) -> Array2<f64> {
        self.mh.dot(x_prime)
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
        let x_prime = self.ma.dot(&self.mx);
        let x = x_prime[[0, 0]];
        let y = x_prime[[1, 0]];
        let den = x * x + y * y;
        let densq = den.sqrt();
        self.mh = array![
            [x / densq, y / densq, 0.0, 0.0],
            [y / den, -x / den, 0.0, 0.0]
        ];
        // Measurement covariance matrix
        self.mr = m.cov.to_owned();

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
        self.mx = array![[temp_x], [temp_y], [0.0], [0.0]];
        // State transistion matrix - linear extrapolation assuming constant velocity
        self.ma[[0, 2]] = Measurement::DT;
        self.ma[[1, 3]] = Measurement::DT;
        // Measurement covariance matrix
        self.mr = m.cov.to_owned();

        Array2::zeros((4, 2))
    }

    fn second(&mut self, m: &mut Measurement) -> Array2<f64> {
        // compute position values from measurements
        // x = r*sin(b)
        let temp_x = m.r * (m.b * std::f64::consts::PI / 180.0).sin();
        // y = r*cos(b)
        let temp_y = m.r * (m.b * std::f64::consts::PI / 180.0).cos();
        // State vector - initialize position values
        self.mx = array![
            [temp_x],
            [temp_y],
            [(temp_x - self.mx[[0, 0]]) / Measurement::DT],
            [(temp_y - self.mx[[1, 0]]) / Measurement::DT]
        ];
        // State covariance matrix
        self.mp[[0, 0]] = 100.0;
        self.mp[[1, 1]] = 100.0;
        self.mp[[2, 2]] = 250.0;
        self.mp[[3, 3]] = 250.0;
        // Measurement covariance matrix
        self.mr = m.cov.to_owned();
        // System error matrix - initialized to zero matrix for first update
        self.mq[[0, 0]] = 20.0;
        self.mq[[1, 1]] = 20.0;
        self.mq[[2, 2]] = 4.0;
        self.mq[[3, 3]] = 4.0;

        Array2::zeros((4, 2))
    }
}

impl Filter {
    fn filter(&mut self, m: &mut Measurement) -> Array2<f64> {
        // Predict mean covariance forward
        let x_prime = self.f(m) + &self.mb;
        let p_prime = self.ma.dot(&self.mp).dot(&self.ma.t()) + &self.mq;

        // CalculateResiduals
        let h_res = self.h(m, &x_prime);    // Extra var to help borrow checker!
        let residual_mean = &m.mz - &h_res;
        let residual_cov = self.mh.dot(&p_prime).dot(&self.mh.t()) + &self.mr;
        // Compute Kalman Gain
        let mk = p_prime.dot(&self.mh.t()).dot(&residual_cov.inv().unwrap());
        // Estimate new State
        self.mx = &x_prime + &mk.dot(&residual_mean);
        // Estimate new Covariance
        self.mp = &p_prime - mk.dot(&self.mh).dot(&p_prime);

        mk
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
    fn new(m: &mut Measurement, filt: &mut Filter, step: u32) -> Self {
        m.get();
        let mk = match step {
            0 => filt.first(m),
            1 => filt.second(m),
            _ => filt.filter(m),
        };

        let est_pos = array![filt.mx[[0, 0]], filt.mx[[1, 0]]];
        let est_vel = array![filt.mx[[2, 0]], filt.mx[[3, 0]]];
        let pos_sigma = array![filt.mp[[0, 0]].sqrt(), filt.mp[[1, 1]].sqrt()];
        let pos_gain = array![mk[[0, 0]], mk[[1, 0]]];
        let vel_gain = array![mk[[2, 0]], mk[[3, 0]]];

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
    let ex = m.iter().map(|i| i.est_pos[0] - i.x);
    let ey = m.iter().map(|i| i.est_pos[1] - i.y);
    let x_pos = m.iter().map(|i| i.est_pos[0]);
    let y_pos = m.iter().map(|i| i.est_pos[1]);
    let x_vel = m.iter().map(|i| i.est_vel[0]);
    let y_vel = m.iter().map(|i| i.est_vel[1]);
    let actual_r = m.iter().map(|i| i.t_r);
    let actual_b = m.iter().map(|i| i.t_b);
    let est_r = m.iter().map(|i| i.r);
    let est_b = m.iter().map(|i| i.b);
    let x_pos_gain = m.iter().map(|i| i.pos_gain[0]);
    let y_pos_gain = m.iter().map(|i| i.pos_gain[1]);
    let x_vel_gain = m.iter().map(|i| i.vel_gain[0]);
    let y_vel_gain = m.iter().map(|i| i.vel_gain[1]);
    let e_x_3sig = m.iter().map(|i| 3.0 * i.pos_sigma[0]);
    let e_y_3sig = m.iter().map(|i| 3.0 * i.pos_sigma[1]);
    let ne_x_3sig = m.iter().map(|i| 3.0 * -i.pos_sigma[0]);
    let ne_y_3sig = m.iter().map(|i| 3.0 * -i.pos_sigma[1]);

    simple_plot::plot!("Actual Range vs Measured Range", actual_r, est_r);
    simple_plot::plot!("Actual Azimuth vs Measured Azimuth", actual_b, est_b);
    simple_plot::plot!("Velocity Estimate On Each Measurement Update", x_vel, y_vel);
    simple_plot::plot!(
        "X Position Estimate Error Containment",
        ex,
        e_x_3sig,
        ne_x_3sig
    );
    simple_plot::plot!(
        "Y Position Estimate Error Containment",
        ey,
        e_y_3sig,
        ne_y_3sig
    );
    // Other useful debugging and understanding plots
    //simple_plot::plot!("Cartesian position data", x_pos, y_pos);
    //simple_plot::plot!("Position Error 3 Sigma", e_x_3sig, e_y_3sig);
    //simple_plot::plot!("Position", ex, ey);
    //simple_plot::plot!("X Kalman Gains", x_pos_gain, x_vel_gain);
    //simple_plot::plot!("Y Kalman Gains", y_pos_gain, y_vel_gain);
    //simple_plot::plot!("Gains", x_pos_gain, x_vel_gain, y_pos_gain, y_vel_gain);
}
