/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define NUM_PARTICLES  1024
#define EPS  0.00001

using namespace std;

//static global random generator
static default_random_engine random_gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  if (!is_initialized) {
    std::cout << "Filter initialized" << std::endl;
    num_particles = NUM_PARTICLES;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
      Particle particle;
      particle.id = i;
      particle.x = dist_x(random_gen);
      particle.y = dist_y(random_gen);
      particle.theta = dist_theta(random_gen);
      particle.weight = 1.0;

      weights.push_back(1.0);
      particles.push_back(particle);
    }
    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    double theta = particles[i].theta;
    double noise_x = dist_x(random_gen);
    double noise_y = dist_y(random_gen);
    double noise_theta = dist_theta(random_gen);

    if (fabs(yaw_rate) < EPS) {
      particles[i].x += velocity * delta_t * cos(theta) + noise_x;
      particles[i].y += velocity * delta_t * sin(theta) + noise_y;
      particles[i].theta += noise_theta;
    } else {
      double upd_theta = theta + delta_t * yaw_rate;
      particles[i].x += velocity / yaw_rate * (sin(upd_theta) - sin(theta)) + noise_x;
      particles[i].y += velocity / yaw_rate * (cos(theta) - cos(upd_theta)) + noise_y;
      particles[i].theta = upd_theta + noise_theta;
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (int i = 0; i < observations.size(); i++) {
    int    min_id = -1;
    double min_distance = numeric_limits<double>::max();

    for (int j = 0; j < predicted.size(); j++) {
      double delta_x = predicted[j].x - observations[i].x;
      double delta_y = predicted[j].y - observations[i].y;
      double distance = delta_x * delta_x + delta_y * delta_y;

      if (distance < min_distance) {
        min_id = j;
        min_distance = distance;
      }
    }
    observations[i].id = min_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  for (int i = 0; i < num_particles; i++) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    vector<LandmarkObs> in_range;
    vector<LandmarkObs> transformed;
    LandmarkObs landmark;

    //Transform landmarks
    for (int j = 0; j < observations.size(); j++) {
      int obs_id = observations[j].id;
      double obs_x = observations[j].x;
      double obs_y = observations[j].y;

      double transformed_x = p_x + obs_x * cos(p_theta) - obs_y * sin(p_theta);
      double transformed_y = p_y + obs_y * cos(p_theta) + obs_x * sin(p_theta);

      landmark.id = obs_id;
      landmark.x = transformed_x;
      landmark.y = transformed_y;

      transformed.push_back(landmark);
    }
    //Find landmarks within sensor range
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      double landmark_x = map_landmarks.landmark_list[j].x_f;
      double landmark_y = map_landmarks.landmark_list[j].y_f;

      double distance_x = landmark_x - p_x;
      double distance_y = landmark_y - p_y;
      double distance = sqrt(distance_x * distance_x + distance_y * distance_y);

      if (distance < sensor_range) {
        landmark.id = landmark_id;
        landmark.x = landmark_x;
        landmark.y = landmark_y;

        in_range.push_back(landmark);
      }
    }
    
    if(!in_range.size()) {
        std::cout << "ERROR: No landmarks in sensor range!" << std::endl;
	return;
    }

    //Associate landmark in range by `id` with observation landmark - update transformed_landmarks
    dataAssociation(in_range, transformed);

    //Update particle weights
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double denominator = sqrt(2.0 * M_PI * std_x * std_y);
    double a_denominator = 2 * std_x * std_x;
    double b_denominator = 2 * std_y * std_y;

    double weight = 1.0;
    for (int j = 0; j < transformed.size(); j++) {
      int obs_id = transformed[j].id;
      double obs_x = transformed[j].x;
      double obs_y = transformed[j].y;
      
      double dx = obs_x - in_range[obs_id].x;
      double dy = obs_y - in_range[obs_id].y;

      double a = dx * dx / a_denominator;
      double b = dy * dy / b_denominator;

      weight *= exp(-(a + b)) / denominator;
    }
    particles[i].weight = weight;
    weights[i] = weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<Particle> new_particles;
  discrete_distribution<int> dd_idx(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; i++) {
    const int idx = dd_idx(random_gen);
    new_particles.push_back(particles[idx]);
  }
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
