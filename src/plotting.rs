#![allow(dead_code, unused_variables)]
use ndarray::Array1;
extern crate plotly;
use plotly::{
        common::{Mode, Title},
        layout::Layout,
        Plot, Scatter,
    };

pub fn new() -> Plot {
    Plot::new()
}

pub fn plotter_xy(plot: &mut Plot, title: &str, label: &str, xa: &[f64], ya: &[f64], scatter: bool) {
    let trace = Scatter::new(xa.to_vec(), ya.to_vec())
        .name(label)
        .mode(if scatter { Mode::Markers } else { Mode::Lines });
    plot.add_trace(trace);
    plot.set_layout(Layout::new().title(Title::new(title)));
}

pub fn plotter_y(plot: &mut Plot, title: &str, label: &str, ya: &[f64], scatter: bool) {
    let xa: Vec<f64> = (0..ya.len()).map(|i| i as f64).collect();
    let trace = Scatter::new(xa.to_vec(), ya.to_vec())
        .name(label)
        .mode(if scatter { Mode::Markers } else { Mode::Lines });
    plot.add_trace(trace);
    plot.set_layout(Layout::new().title(Title::new(title)));
}

pub fn plotter_pair(plot: &mut Plot, title: &str, label: &str, states: &[Array1<f64>], scatter: bool) {
    let xa: Vec<f64> = states.iter().map(|i| i[0]).collect();
    let ya: Vec<f64> = states.iter().map(|i| i[1]).collect();
    let trace = Scatter::new(xa.to_vec(), ya.to_vec())
        .name(label)
        .mode(if scatter { Mode::Markers } else { Mode::Lines });
    plot.add_trace(trace);
    plot.set_layout(Layout::new().title(Title::new(title)));
}

