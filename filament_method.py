

import numpy as np
import pandas as pd
from scipy.special import ellipk, ellipe
from matplotlib import pyplot as plt



from plot_results import plot_force, dynamic_func, calculate_r2_rmse

# Define constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)

def k_constant(a, b, z):
    """
    Calculate the constant k for elliptic integrals.
    a, b: Radii of the two circular segments
    z: Distance between the two segments
    """
    return np.sqrt(4 * a * b / ((a + b)**2 + z**2))

def segment_current(turns, total_current, nr, na):
    """
    Calculate the current for each segment based on the total current and number of segments.
    total_current: Total current passing through the coil
    nr: Number of radial segments
    na: Number of axial segments
    """
    curr = (total_current * turns) / (nr * na)
    return curr

def k_prime(r_k, r_j, z_il):
    """
    Calculate the constant k' for elliptic integrals.
    """
    return np.sqrt(4 * r_k * r_j / ((r_k + r_j)**2 + z_il**2))

def current_mag_coil(J, H, W):
    """
    Calculate the current for equivalent solenoid of a permanent magnet
    """
    MU_0 = 4 * np.pi * 1e-7
    return (J * H)/(MU_0 * W)
def force_between_segments_SMES(i1, i2, r_k, r_j, z_il):
    """
    Calculate the force between two segments using elliptic integrals.
    """
    k_prime_val = k_prime(r_k, r_j, z_il)
    prefactor = (MU_0 * i1 * i2 * z_il * k_prime_val) / (2 * np.sqrt(r_k * r_j) * (1 - k_prime_val**2))

    # Elliptic integrals of the first (K) and second (E) kind
    K_k = ellipk(k_prime_val**2)
    E_k = ellipe(k_prime_val**2)

    force =  prefactor * ((1 - k_prime_val**2) * K_k - (1 - (k_prime_val**2 / 2)) * E_k)

    return force

def force_between_segments_OMES(i1, i2, r_k, r_j, z_il):
    """
    Calculate the force between two segments using elliptic integrals.
    """
    k_prime_val = k_prime(r_k, r_j, z_il)
    prefactor = (MU_0 * i1 * i2 * z_il * k_prime_val) / (2 * np.sqrt(r_k * r_j) * (1 - k_prime_val**2))

    # Elliptic integrals of the first (K) and second (E) kind
    K_k = ellipk(k_prime_val**2)
    E_k = ellipe(k_prime_val**2)

    force =  - prefactor * ((1 - k_prime_val**2) * K_k - (1 - (k_prime_val**2 / 2)) * E_k)

    return force

def force_between_segments_SMES(i1, i2, r_k, r_j, z_il):
    """
    Calculate the force between two segments using elliptic integrals.
    """
    k_prime_val = k_prime(r_k, r_j, z_il)
    prefactor = (MU_0 * i1 * i2 * z_il * k_prime_val) / (2 * np.sqrt(r_k * r_j) * (1 - k_prime_val**2))

    # Elliptic integrals of the first (K) and second (E) kind
    K_k = ellipk(k_prime_val**2)
    E_k = ellipe(k_prime_val**2)

    force =  prefactor * ((1 - k_prime_val**2) * K_k - (1 - (k_prime_val**2 / 2)) * E_k)

    return force

def total_force_S_MLE(seq, turn_1, turn_2, i1, i2, nr1, nr2, na1, na2, r1_min, r1_max, r2_min, r2_max, z, a1, a2):
    """
    Calculate the total force between two coils by summing forces between segments.
    """

    layer = len(seq)
    layer_a1, layer_a2 = 0, 0
    current_a1 = 1
    current_a2 = 1
    base_na1 = na1
    base_na2 = na2

    turn_1 = layer * turn_1
    turn_2 = layer * turn_2

    na1 = layer * na1
    na2 = layer * na2

    a1 = layer * a1
    a2 = layer * a2

    # print( f'na1: {na1}, na2: {na2}, base_na1: {base_na1}, base_na2: {base_na2}')


    # Radial thicknesses
    b1 = r1_max - r1_min
    b2 = r2_max - r2_min

    # Radial positions of segments
    r1_positions = [r1_min + (0.5 + j) * (b1 / nr1) for j in range(nr1)]
    r2_positions = [r2_min + (0.5 + k) * (b2 / nr2) for k in range(nr2)]

    # Calculate the segment currents
    i1_segment = segment_current(turn_1, i1, nr1, na1)
    i2_segment = segment_current(turn_2, i2, nr2, na2)

    total_force = 0.0

    # Loop over all pairs of segments
    for r_j in r1_positions:
        for r_k in r2_positions:
            # Loop over axial segments
            for i in range(na1):
                if i % base_na1 == 0:
                  i1_segment = (current_a1/seq[layer_a1]) * i1_segment
                  current_a1 = seq[layer_a1]
                  layer_a1 += 1
                for l in range(na2):
                    if l % base_na2 == 0:
                      i2_segment = (current_a2/seq[layer_a2]) * i2_segment
                      # print('changed a2')
                      # print(i2_segment)
                      current_a2 = seq[layer_a2]
                      layer_a2 += 1
                    # Calculate z_il using the height of coils and the z distance
                    z_il = z - (a2 / 2 + a1 / 2) + (0.5 + i) * (a1 / na1) + (0.5 + l) * (a2 / na2)

                    # Add up the force between these two segments
                    total_force += force_between_segments_SMES(i1_segment, i2_segment, r_j, r_k, z_il)
                # print(layer_a2)
                layer_a2 = 0
            # print(layer_a1)
            layer_a1 = 0
    # print(i1, i1_segment)
    # print(i2, i2_segment)
    return total_force

def total_force_O_MLE(seq, turn_1, turn_2, i1, i2, nr1, nr2, na1, na2, r1_min, r1_max, r2_min, r2_max, z, a1, a2):
    """
    Calculate the total force between two coils by summing forces between segments.
    """

    layer = len(seq)
    layer_a1, layer_a2 = 0, 0
    current_a1 = 1
    current_a2 = 1
    base_na1 = na1
    base_na2 = na2

    turn_1 = layer * turn_1
    turn_2 = layer * turn_2

    na1 = layer * na1
    na2 = layer * na2

    a1 = layer * a1
    a2 = layer * a2

    # print( f'na1: {na1}, na2: {na2}, base_na1: {base_na1}, base_na2: {base_na2}')


    # Radial thicknesses
    b1 = r1_max - r1_min
    b2 = r2_max - r2_min

    # Radial positions of segments
    r1_positions = [r1_min + (0.5 + j) * (b1 / nr1) for j in range(nr1)]
    r2_positions = [r2_min + (0.5 + k) * (b2 / nr2) for k in range(nr2)]

    # Calculate the segment currents
    i1_segment = segment_current(turn_1, i1, nr1, na1)
    i2_segment = segment_current(turn_2, i2, nr2, na2)

    total_force = 0.0

    # Loop over all pairs of segments
    for r_j in r1_positions:
        for r_k in r2_positions:
            # Loop over axial segments
            for i in range(na1):
                if i % base_na1 == 0:
                  i1_segment = (current_a1/seq[layer_a1]) * i1_segment
                  current_a1 = seq[layer_a1]
                  layer_a1 += 1
                for l in range(na2):
                    if l % base_na2 == 0:
                      i2_segment = (current_a2/seq[layer_a2]) * i2_segment
                      # print('changed a2')
                      # print(i2_segment)
                      current_a2 = seq[layer_a2]
                      layer_a2 += 1
                    # Calculate z_il using the height of coils and the z distance
                    z_il = z - (a2 / 2 + a1 / 2) + (0.5 + i) * (a1 / na1) + (0.5 + l) * (a2 / na2)

                    # Add up the force between these two segments
                    total_force += force_between_segments_OMES(i1_segment, i2_segment, r_j, r_k, z_il)
                # print(layer_a2)
                layer_a2 = 0
            # print(layer_a1)
            layer_a1 = 0
    # print(i1, i1_segment)
    # print(i2, i2_segment)
    return total_force

def force_SMES(sequence, turn_coil, turn_mag, i_coil, i_mag, v, u, w, r_coil_i, r_coil_o, r_mag_i, r_mag_o, h, range_l, range_h, step):

  outp = []
  for i in range(-range_l, range_h, step):
    z_distance = i/1000
    total_force_1 = total_force_S_MLE(sequence, turn_coil, turn_mag, -i_coil, i_mag, v, 1, u, w, r_coil_i, r_coil_o, r_mag_o, r_mag_o, z_distance, h, h)
    total_force_2 = total_force_S_MLE(sequence, turn_coil, turn_mag, -i_coil, -i_mag, v, 1, u, w, r_coil_i, r_coil_o, r_mag_i, r_mag_i, z_distance, h, h)
    total_force = total_force_1 + total_force_2
    # print(f"Total Magnetic Force between Coils {z_distance}: {total_force:.4e} N")
    outp.append((z_distance, total_force))
  return outp

def force_OMES(sequence, turn_coil, turn_mag, i_coil, i_mag, v, u, w, r_coil_i, r_coil_o, r_mag_i, r_mag_o, h, range_l, range_h, step):

  outp = []
  for i in range(-range_l, range_h, step):
    z_distance = i/1000
    total_force_1 = total_force_O_MLE(sequence, turn_coil, turn_mag, -i_coil, i_mag, v, 1, u, w, r_coil_i, r_coil_o, r_mag_o, r_mag_o, z_distance, h, h)
    total_force_2 = total_force_O_MLE(sequence, turn_coil, turn_mag, -i_coil, -i_mag, v, 1, u, w, r_coil_i, r_coil_o, r_mag_i, r_mag_i, z_distance, h, h)
    total_force = total_force_1 + total_force_2
    # print(f"Total Magnetic Force between Coils {z_distance}: {total_force:.4e} N")
    outp.append((z_distance, total_force))
  return outp



def main():
  j = 0.347

  w = 50
  u = 15
  v = 18

  #height of the coil and magnet
  h =  0.0125
  #current on the coil
  i_coil = 3
  #magnet radius inside and outside
  r_mag_i = 0.005
  r_mag_o = 0.014
  #coil radius inside and outside
  r_coil_i = 0.017
  r_coil_o = 0.025
  #Number of turns/conductor
  turn_coil = 30
  turn_mag = w
  #Conversion of permanent magnet into solenoid
  i_mag = current_mag_coil(j, h, w)

  z_distance = 0.0125

  seq = [[1], [1,1], [1, 1, 1], [1, 1, 1, 1], [1,1,1,1,1], [1,1,1,1,1,1]]
  i_coil_s = [3, 3.5, 4, 4.5]
  for i_coil in i_coil_s:
    for po in range(len(seq)):
      sequence = seq[po]
      output = force_SMES(sequence, turn_coil, turn_mag, i_coil, i_mag, v, u, w, r_coil_i, r_coil_o, r_mag_i, r_mag_o, h, 60, 61, 2)
      plot_force(output)
      output_df = pd.DataFrame(output, columns=['origin_z', 'Force_z'])
      filename = f"{po}_SMES_Layer_{i_coil}_.csv"
      output_df.to_csv(filename, index = False)


    
if __name__ == "__main__":
  main()