#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_must_use)]
use ndarray::prelude::*;
use ndarray_inverse::*;
use kalman::plotting::*;
use rand::prelude::*;
use rand_distr::Normal;
use plotly::Plot;

#[derive(Debug)]
struct Measurement {
    x: f64,
    y: f64,
    t_d: f64,
    t_r: f64,
    t_b: f64,
    r: f64,
    b: f64,
    z: Array1<f64>,
}

impl Measurement {
    // Bearing sensor standard deviation = 9 milliradians (in degrees)
    const SIG_B: f64 = 0.009 * 180.0 / std::f64::consts::PI;
    const MAX_VAR: f64 = 0.25;
    // Range sensor Standard Deviation = 30 meters
    const SIG_R: f64 = 30.0;
    const X_VEL: f64 = 22.0;
    const Y_VEL: f64 = 0.0;

    fn new() -> Self {
        Self {
            x: 2900.0,
            y: 2900.0,
            t_d: 0.0,
            t_r: 0.0,
            t_b: 0.0,
            r: 0.0,
            b: 0.0,
            z: array![0.0, 0.0],
        }
    }

    fn get(&mut self, filt: &mut Filter, dt: f64) {
        let mut rng = rand::thread_rng();
        //let mut rng = Pcg64::from_seed([1; 32]);
        let normal = Normal::new(1.0, Measurement::MAX_VAR).unwrap();
        // Compute the actual new x and y position data with all of the
        // velocity motion is in the x direction i.e. linear motion
        self.x += dt * Measurement::X_VEL * normal.sample(&mut rng);
        self.y += dt * Measurement::Y_VEL * normal.sample(&mut rng);
        // Compute range and bearing (azimuth)
        (self.t_d, self.t_r, self.t_b) = car2pol_deg(self.x, self.y);

        let normal = Normal::new(0.0, Measurement::MAX_VAR).unwrap();
        // Compute the error for each measurement
        // By taking the max between of the defined standard deviation and
        // the randomly generated normal error, it guarantees an error
        let t_sig_b = if normal.sample(&mut rng) > 0.0 {
            (Measurement::SIG_B * normal.sample(&mut rng)).max(Measurement::SIG_B * Measurement::MAX_VAR)
        } else {
            (Measurement::SIG_B * normal.sample(&mut rng)).min(-Measurement::SIG_B * Measurement::MAX_VAR)
        };
        let t_sig_r = if normal.sample(&mut rng) > 0.0 {
            (Measurement::SIG_R * normal.sample(&mut rng)).max(Measurement::SIG_R * Measurement::MAX_VAR)
        } else {
            (Measurement::SIG_R * normal.sample(&mut rng)).min(-Measurement::SIG_R * Measurement::MAX_VAR)
        };
        // Save off the measurement values for bearing and range as a Function
        // of the true value + the error generated above
        self.r = self.t_r + t_sig_r;
        self.b = self.t_b + t_sig_b;

        self.z = array![self.r, self.b];

        // Later becomes R - Measurement Covariance Matrix
        filt.R = Array2::from_diag(&array![t_sig_r.powi(2), t_sig_b.powi(2)]);
    }
}

#[derive(Debug)]
#[allow(non_snake_case)]
struct Filter {
    x: Array1<f64>,
    P: Array2<f64>,
    z: Array1<f64>, // z is the Measurement
    A: Array2<f64>,
    B: Array2<f64>,
    H: Array2<f64>,
    R: Array2<f64>,
    Q: Array2<f64>,
    // K is internal
    dim_x: usize,
    dim_z: usize,
    dim_u: usize,
}

trait FilterNew {
    fn new(dim_x: usize, dim_z: usize, dim_u: usize) -> Self;
}

#[allow(non_snake_case)]
impl FilterNew for Filter {
    /**
     * dim_x: Number of inputs, e.g. dimensions * order
     * dim_z: Number of measurement inputs, e.g. dimensions as in position
     * dim_u: Control inputs - zero, there are none
     */
    fn new(dim_x: usize, dim_z: usize, dim_u: usize) -> Self {
        // System error matrix - initialized to zero matrix for first update
        Self {
            x: array![2900.0, 2900.0, 0.0, 0.0],
            P: Array2::<f64>::from_diag(&array![100.0, 100.0, 250.0, 250.0]),
            z: Array1::<f64>::zeros(dim_z),
            A: Array2::eye(dim_x),
            B: Array2::<f64>::eye(dim_u),
            H: Array2::<f64>::zeros((dim_z, dim_x)),
            R: Array2::<f64>::zeros((dim_z, dim_z)),
            Q: Array2::<f64>::from_diag(&array![20.0, 20.0, 4.0, 4.0]),
            dim_x,
            dim_z,
            dim_u,
        }
    }
}

/*
trait FilterLinearFunctions {
    fn fx(&mut self, m: &mut Measurement) -> Array2<f64>;
    fn hx(&mut self, m: &mut Measurement) -> Array2<f64>;
}

impl FilterLinearFunctions for Filter {
    fn fx(&mut self) -> Array2<f64> {
        self.A.dot(&self.x)
    }

    fn hx(&mut self, x_prime: &Array2<f64>) -> Array2<f64> {
        self.H.dot(x_prime)
    }
}
*/

trait FilterFunctions {
    fn fx(&mut self, u: &Array1<f64>, dt: f64) -> Array1<f64>;
    fn hx(&self, z: &Array1<f64>, x_prime: &Array1<f64>) -> Array1<f64>;
}

impl FilterFunctions for Filter {
    fn fx(&mut self, u: &Array1<f64>, _dt: f64) -> Array1<f64> {
        // Form state to measurement transition matrix
        let x_prime = if self.dim_u > 0 {
            self.A.dot(&self.x) + self.B.dot(u)
        } else {
            self.A.dot(&self.x)
        };
        let x = x_prime[0];
        let y = x_prime[1];
        let (d, r, b) = car2pol_deg(x, y);
        // Sensor reading error covariance Matrix
        self.H = array![[x / r, y / r, 0.0, 0.0],
                        [y / d, -x / d, 0.0, 0.0]];

        x_prime
    }

    fn hx(&self, z: &Array1<f64>, x_prime: &Array1<f64>) -> Array1<f64> {
        // Convert the predicted cartesian state to polar range and bearing
        let (_, r, b) = car2pol_deg(x_prime[0], x_prime[1]);

        array![r, b]
    }
}

#[allow(non_snake_case)]
impl Filter {
    // Predict mean and covariance forward
    fn predict(&mut self, u: &Array1<f64>) -> (Array1<f64>, Array2<f64>) {
        let x_prime = self.fx(u, 1.0);
        let p_prime = self.A.dot(&self.P).dot(&self.A.t()) + &self.Q;

        (x_prime, p_prime)
    }

    // Compute Kalman Gain
    fn gain(&self, p_prime: &Array2<f64>) -> Array2<f64> {
        let residual_cov = self.H.dot(p_prime).dot(&self.H.t()) + &self.R;
        
        p_prime.dot(&self.H.t().dot(&residual_cov.inv().unwrap()))
    }

    // Calculate Residuals and measurement update
    fn update(&mut self, K: &Array2<f64>, z: &Array1<f64>, x_prime: &Array1<f64>, p_prime: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
        let h_res = self.hx(z, x_prime);    // Extra var to help borrow checker!
        let residual_mean = z - &h_res;
        // Estimate new State
        let updated_x  = x_prime + &K.dot(&residual_mean);
        // Estimate new Covariance
        let updated_P = p_prime - &K.dot(&self.H).dot(p_prime);
        let updated_P = (&updated_P + &updated_P.t()) / 2.0;  // stability

        (updated_x, updated_P)
    }

    fn filter(&mut self, u: &Array1<f64>, z: &Array1<f64>, dt: f64) -> Array2<f64> {
        set_A(&mut self.A, dt, &[2]);
        let (x_prime, p_prime) = self.predict(u);
        let K = self.gain(&p_prime);

        (self.x, self.P) = self.update(&K, z, &x_prime, &p_prime);

        K
    }
}

/// Set A time deltas give dimesions that change
fn set_A(A: &mut Array2<f64>, dt: f64, pos: &[usize]) {
    let d = A.raw_dim()[0]; // Matrix must be square
    for &i in pos {
        A[[0, i]] = dt;
        A[[1, d - i + 1]] = dt;
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
    fn new(m: &mut Measurement, filt: &mut Filter, dt: f64) -> Self {
        m.get(filt, dt);
        let mk = filt.filter(&array![], &m.z, dt);

        let est_pos = array![filt.x[0], filt.x[1]];
        let est_vel = array![filt.x[2], filt.x[3]];
        let pos_sigma = array![filt.P[[0, 0]].sqrt(), filt.P[[1, 1]].sqrt()];
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

fn car2pol_deg(x: f64, y:f64) -> (f64, f64, f64) {
    let d = x * x + y * y;
    let r = d.sqrt();
    let b = x.atan2(y);

    (d, r, b.to_degrees())
}

#[allow(unused_variables)]
fn main() {
    let dt = 1.0;
    let mut measurement = Measurement::new();
    let mut filter = Filter::new(4, 2, 0);
    let m: Vec<Measure> = (0 .. 30)
        .map(|k| Measure::new(&mut measurement, &mut filter, dt))
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

    let mut plot = Plot::new();
    plotter_y(&mut plot, "Ranges", "Actual", &actual_r, false);
    plotter_y(&mut plot, "Ranges", "Measured", &est_r, true);
    plot.show();

    let mut plot = Plot::new();
    plotter_y(&mut plot, "Azimuths", "Actual", &actual_b, false);
    plotter_y(&mut plot, "Azimuths", "Measured", &est_b, true);
    plot.show();

    let mut plot = Plot::new();
    plotter_y(&mut plot, "Measurement Updates", "X Velocity", &x_vel, false);
    plotter_y(&mut plot, "Measurement Updates", "Y Velocity", &y_vel, false);
    plot.show();

    let mut plot = Plot::new();
    plotter_y(&mut plot, "X Position Esimate Error Containment", "Esimate", &ex, true);
    plotter_y(&mut plot, "X Position Esimate Error Containment", "Positive 3 sigma", &e_x_3sig, false);
    plotter_y(&mut plot, "X Position Esimate Error Containment", "Negative 3 sigma", &ne_x_3sig, false);
    plot.show();

    let mut plot = Plot::new();
    plotter_y(&mut plot, "Y Position Esimate Error Containment", "Esimate", &ey, true);
    plotter_y(&mut plot, "Y Position Esimate Error Containment", "Positive 3 sigma", &e_y_3sig, false);
    plotter_y(&mut plot, "Y Position Esimate Error Containment", "Negative 3 sigma", &ne_y_3sig, false);
    plot.show();
}
