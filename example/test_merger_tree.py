import numpy as np
import matplotlib.pyplot as plt
from toolscosmo import merger_trees

# Define the parameters for the merger tree
final_mass = 1e12  # M_sun, e.g., a Milky Way-like halo
final_redshift = 0
max_redshift = 6.0
mass_resolution = 1e8 # M_sun

# Create a MergerTree instance
tree_generator = merger_trees.MergerTree(
    M0=final_mass,
    z0=final_redshift,
    z_max=max_redshift,
    M_res=mass_resolution
)

# Run the simulation
tree_generator.run(verbose=True)

# Display the results
print("\n--- Main Progenitor Branch ---")
print(tree_generator.main_branch_history.head())

print("\n--- Accreted Subhalos ---")
if not tree_generator.subhalo_history.empty:
    print(tree_generator.subhalo_history.head())
else:
    print("No subhalos were resolved in this run.")

# Plot the results using the new method
tree_generator.plot_merger_history()
