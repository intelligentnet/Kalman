use ndarray::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;
use ndarray_inverse::Inverse;

struct Measurement {
    current_position: f64,
    current_velocity: f64,
}

impl Measurement {
    fn new() -> Measurement {
        Measurement {
            current_position: 0.0,
            current_velocity: 60.0,
        }
    }

    fn get(&mut self) -> (f64, f64, f64) {
        let dt = 0.1;
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 8.0).unwrap();
        let w = normal.sample(&mut rng);
        let v = normal.sample(&mut rng);

        let z = self.current_position + self.current_velocity * dt + v;

        self.current_position = z - v;
        self.current_velocity = 60.0 + w;

        (z, self.current_position, self.current_velocity)
    }
}

struct Filter {
    x: Array2<f64>,
    p: Array2<f64>,
    a: Array2<f64>,
    b: Array2<f64>,
    h: Array2<f64>,
    ht: Array2<f64>,
    r: f64,
    q: Array2<f64>,
}

impl Filter {
    fn new() -> Filter {
        let dt = 0.1;
        let x = array![[0.0], [20.0]];
        let p = array![[5.0, 0.0], [0.0, 5.0]];
        let a = array![[1.0, dt], [0.0, 1.0]];
        let b = array![[0.0, 0.0], [0.0, 0.0]];
        let h = array![[1.0, 0.0]];
        let ht = h.t().to_owned();
        let r = 10.0;
        let q = array![[1.0, 0.0], [0.0, 3.0]];

        Filter {
            x,
            p,
            a,
            b,
            h,
            ht,
            r,
            q,
        }
    }

    fn linear_filter(&mut self, z: f64) -> Array2<f64> {
        let f = |x: &Array2<f64>, y: &Array2<f64>| x.dot(y);
        self.filter(z, &f, &f)
    }

    fn filter(
        &mut self,
        z: f64,
        f: &dyn Fn(&Array2<f64>, &Array2<f64>) -> Array2<f64>,
        h: &dyn Fn(&Array2<f64>, &Array2<f64>) -> Array2<f64>,
    ) -> Array2<f64> {
        // Predict mean forward
        let x_prime = f(&self.a, &self.x) + &self.b;
        // Predict Covariance Forward
        let p_prime = self.a.dot(&self.p).dot(&self.a.t()) + &self.q;

        // Compute Kalman Gain
        let s = self.h.dot(&p_prime).dot(&self.ht) + self.r;
        let k = p_prime.dot(&self.ht).dot(&s.inv().unwrap());

        // Estimate new State
        self.x = &x_prime + &k * (z - h(&self.h, &x_prime));
        // Estimate new Covariance
        self.p = &p_prime - k.dot(&self.h).dot(&p_prime);

        k
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct Measure {
    time: i32,
    pos: f64,
    est_pos: f64,
    est_vel: f64,
    dif_pos: f64,
    est_dif_pos: f64,
    pos_bound3sigma: f64,
    pos_gain: f64,
    vel_gain: f64,
}

impl Measure {
    fn new(measurement: &mut Measurement, filter: &mut Filter, time: i32) -> Measure {
        let z = measurement.get();
        let k = filter.linear_filter(z.0);

        let pos = z.0;
        let dif_pos = z.0 - z.1;
        let est_pos = filter.x[[0, 0]];
        let est_dif_pos = est_pos - z.1;
        let est_vel = filter.x[[1, 0]];
        let pos_var = filter.p[[0, 0]];
        let pos_bound3sigma = 3.0 * pos_var.sqrt();
        let pos_gain = k[[0, 0]];
        let vel_gain = k[[1, 0]];

        Measure {
            time,
            pos,
            est_pos,
            est_vel,
            dif_pos,
            est_dif_pos,
            pos_bound3sigma,
            pos_gain,
            vel_gain,
        }
    }
}

fn main() {
    let mut measurement = Measurement::new();
    let mut filter = Filter::new();
    let m: Vec<Measure> = (1..300)
        .map(|k| Measure::new(&mut measurement, &mut filter, k))
        .collect();
    let pos: Vec<f64> = m.iter().map(|i| i.pos).collect();
    let est_diff_pos: Vec<f64> = m.iter().map(|i| i.est_dif_pos).collect();
    let est_vel: Vec<f64> = m.iter().map(|i| i.est_vel).collect();
    let pos_gain: Vec<f64> = m.iter().map(|i| i.pos_gain).collect();
    let vel_gain: Vec<f64> = m.iter().map(|i| i.vel_gain).collect();

    simple_plot::plot!("Actual position data", pos);
    simple_plot::plot!(
        "Estimated Position difference and velocity",
        est_diff_pos,
        est_vel
    );
    simple_plot::plot!("Kalman Gains", pos_gain, vel_gain);
}
