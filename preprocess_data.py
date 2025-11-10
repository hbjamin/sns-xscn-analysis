#!/usr/bin/env python
"""
Store reconstructed energy and direction in compressed .npz files
for all events from specified root files with Nhit > 0

Updated: Now uses relative paths from config
"""

import numpy as np
import uproot
import awkward as ak
import glob
from datetime import datetime

# Import config for paths
import config as cfg

# Energy reconstruction factors for total charge
ALPHA_WATER = cfg.ALPHA_WATER
ALPHA_1WBLS = cfg.ALPHA_1WBLS

# Channel configurations
# Note: These root file paths still need to be configured based on your system
# They are left as absolute paths since they reference data outside the project
CHANNELS = {
    'water': {
        'alpha': ALPHA_WATER, 
        'channels': {
            'nueGa71': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/gallium-rat-water/task_*.ntuple.root',
            'eES': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/es-rat-water/task_*.ntuple.root',
            'cosmics': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/cosmics-rat-water-merged/task_*.root',
            'nueO16': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/newton-rat-water/task_*.ntuple.root',
            'neutrons_0ft': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/neutrons-rat-water-0ft/task_*.root',
            'neutrons_1ft': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/neutrons-rat-water-1ft/task_*.root',
            'neutrons_3ft': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/neutrons-rat-water-3ft/task_*.root',
        },
        'reverse_direction': {'nueO16': True}
    },
    '1wbls': {
        'alpha': ALPHA_1WBLS,
        'channels': {
            'nueGa71': None, # need to make this 
            'eES': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/es-rat-1wbls/task_*.ntuple.root',
            'cosmics': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/cosmics-rat-1wbls/task_*.ntuple.root',
            'nueO16': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/newton-rat-1wbls/task_*.ntuple.root',
            'neutrons_0ft': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/neutrons-rat-1wbls-0ft/task_*.root',
            'neutrons_1ft': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/neutrons-rat-1wbls-1ft/task_*.root',
            'neutrons_3ft': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/neutrons-rat-1wbls-3ft/task_*.root',
        },
        'reverse_direction': {'nueO16': True}
    }
}


def preprocess_channel(file_pattern, alpha, output_path, reverseXdir=False):
    """
    Process channel data and save to compressed npz file.
    
    Inputs 
    ----------
    file_pattern : str : Glob pattern for root files
    alpha : float : Energy calibration constant
    output_path : Path : Where to save the npz file
    reverseXdir : bool : Whether to reverse direction
        
    Outputs 
    -------
    ntrig : int : Number of triggered events
    nsim : int : Number of simulated events
    """
    files = sorted(glob.glob(file_pattern))
    
    if len(files) == 0:
        print(f"  ERROR: No files found matching {file_pattern}")
        return 0, 0
    
    print(f"  Processing {len(files)} files...")
    start_time = datetime.now()
    
    all_energy= []
    all_mcke = []
    all_direction = []
    total_nsim = 0
    
    for i, file in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{len(files)} files")
        
        try:
            f = uproot.open(file)
            tree = f['output']
            
            # Load only what we need - USE NHIT INSTEAD OF DIGITCHARGE!
            vmask = tree['validposition_quadfitter'].array(library='np')
            fitu = tree['u_fitdirectioncenter_0p5_quad'].array(library='np')
            nhit = tree['digitNhits'].array(library='np')  # MUCH FASTER than digitCharge!
            mcke = tree['mcke'].array(library='np') # only use this for neutrons
            tcharge = ak.sum(ak.Array(tree['digitCharge'].array(library='ak')), axis=1)
            tcharge = ak.to_numpy(tcharge)
            
            total_nsim += len(nhit)
            
            # Apply cuts - nhit > 0 is much faster than tcharge > 0
            nevents = min(len(vmask), len(fitu), len(nhit), len(tcharge))
            vmask = vmask[:nevents] == 1
            nhit = nhit[:nevents]
            mcke = mcke[:nevents]
            tcharge = tcharge[:nevents]
            fitu = fitu[:nevents]
            
            nhit_mask = nhit > 0
            mask = vmask & nhit_mask
            
            # Extract energy and direction
            energy = tcharge[mask] * alpha
            direction = fitu[mask]
            mcke = mcke[mask]
            
            if reverseXdir:
                direction = direction * -1
            
            all_energy.append(energy)
            all_direction.append(direction)
            all_mcke.append(mcke)
            
        except Exception as e:
            print(f"    WARNING: Failed to process {file}: {e}")
            continue
    
    if len(all_energy) == 0:
        print(f"  ERROR: No data extracted from any files")
        return 0, 0
    
    # Concatenate and save
    energy = np.concatenate(all_energy)
    direction = np.concatenate(all_direction)
    mcke = np.concatenate(all_mcke)
    ntrig = len(energy)
    nsim = total_nsim
    
    # Save to compressed .npz
    np.savez_compressed(output_path, energy=energy, direction=direction, mcke=mcke, ntrig=ntrig, nsim=nsim)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    file_size_mb = output_path.stat().st_size / 1e6
    
    print(f"  ✓ Saved {ntrig:,} events to {output_path}")
    print(f"    File size: {file_size_mb:.1f} MB")
    print(f"    Processing time: {elapsed:.1f} seconds")
    print(f"    Efficiency: {ntrig}/{nsim} = {100*ntrig/nsim:.1f}%")
    
    return ntrig, nsim

def main():
    """Main preprocessing pipeline."""
    
    print("=" * 80)
    print("SNS DATA PREPROCESSING")
    print("=" * 80)
    print(f"Output directory: {cfg.PREPROCESSED_DIR}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure output directory exists
    cfg.PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each detector
    for detector_name, detector_config in CHANNELS.items():
        print(f"\n{'='*80}")
        print(f"DETECTOR: {detector_name.upper()}")
        print(f"{'='*80}")
        
        alpha = detector_config['alpha']
        
        # Process regular channels
        print(f"\nProcessing regular channels:")
        for channel_name, file_pattern in detector_config['channels'].items():
            if file_pattern is None:
                print(f"\n{channel_name}: SKIPPED (file path not available)")
                continue
            
            print(f"\n{channel_name}:")
            
            output_filename = f"{detector_name}_{channel_name}.npz"
            output_path = cfg.PREPROCESSED_DIR / output_filename
            
            # Check if already processed
            if output_path.exists():
                print(f"  Already processed: {output_path}")
                print(f"  Delete this file if you want to reprocess")
                continue
            
            reverse = detector_config['reverse_direction'].get(channel_name, False)
            ntrig, nsim = preprocess_channel(file_pattern, alpha, output_path, reverseXdir=reverse)
        
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nPreprocessed files saved to: {cfg.PREPROCESSED_DIR}")
    print("\nYou can now run the analysis with these files (should be ~100x faster!)")
    
    # List all created files
    print(f"\nCreated files:")
    npz_files = sorted(cfg.PREPROCESSED_DIR.glob("*.npz"))
    total_size_mb = 0
    for npz_file in npz_files:
        size_mb = npz_file.stat().st_size / 1e6
        total_size_mb += size_mb
        print(f"  {npz_file.name:40s} {size_mb:>8.1f} MB")
    print(f"\nTotal size: {total_size_mb:.1f} MB")


if __name__ == "__main__":
    import os  # Import here for file operations
    main()
