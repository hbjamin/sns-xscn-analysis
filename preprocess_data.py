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
            'cosmics': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/cosmics-rat-water/task_*.ntuple.root',
            'nueO16': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/newton-rat-water/task_*.ntuple.root',
        },
        'neutrons': {
            '0ft': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/neutrons/water1mil.0ft.root',
            '1ft': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/neutrons/water1mil.1ft.root',
            '3ft': '/nfs/disk1/users/bharris/eos/sim/outputs/sns/neutrons/water1mil.3ft.root',
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
        },
        'neutrons': {
            '0ft': '/home/rbonv/wbls100k.0ft.root',
            '1ft': '/home/rbonv/wbls100k.1ft.root',
            '3ft': '/home/rbonv/wbls100k.3ft.root',
        },
        'reverse_direction': {'nueO16': True}
    }
}


def preprocess_channel(file_pattern, alpha, output_path, reverseXdir=False):
    """
    Process channel data and save to compressed npz file.
    
    Parameters
    ----------
    file_pattern : str
        Glob pattern for root files
    alpha : float
        Energy calibration constant
    output_path : Path
        Where to save the npz file
    reverseXdir : bool
        Whether to reverse direction
        
    Returns
    -------
    ntrig : int
        Number of triggered events
    nsim : int
        Number of simulated events
    """
    files = sorted(glob.glob(file_pattern))
    
    if len(files) == 0:
        print(f"  ERROR: No files found matching {file_pattern}")
        return 0, 0
    
    print(f"  Processing {len(files)} files...")
    start_time = datetime.now()
    
    all_energy = []
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
            tcharge = ak.sum(ak.Array(tree['digitCharge'].array(library='ak')), axis=1)
            tcharge = ak.to_numpy(tcharge)
            
            total_nsim += len(nhit)
            
            # Apply cuts - nhit > 0 is much faster than tcharge > 0
            nevents = min(len(vmask), len(fitu), len(nhit), len(tcharge))
            vmask = vmask[:nevents] == 1
            nhit = nhit[:nevents]
            tcharge = tcharge[:nevents]
            fitu = fitu[:nevents]
            
            nhit_mask = nhit > 0
            mask = vmask & nhit_mask
            
            # Extract energy and direction
            energy = tcharge[mask] * alpha
            direction = fitu[mask]
            
            if reverseXdir:
                direction = direction * -1
            
            all_energy.append(energy)
            all_direction.append(direction)
            
        except Exception as e:
            print(f"    WARNING: Failed to process {file}: {e}")
            continue
    
    if len(all_energy) == 0:
        print(f"  ERROR: No data extracted from any files")
        return 0, 0
    
    # Concatenate and save
    energy = np.concatenate(all_energy)
    direction = np.concatenate(all_direction)
    ntrig = len(energy)
    nsim = total_nsim
    
    # Save to compressed .npz
    np.savez_compressed(output_path, energy=energy, direction=direction, ntrig=ntrig, nsim=nsim)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    file_size_mb = output_path.stat().st_size / 1e6
    
    print(f"  ✓ Saved {ntrig:,} events to {output_path}")
    print(f"    File size: {file_size_mb:.1f} MB")
    print(f"    Processing time: {elapsed:.1f} seconds")
    print(f"    Efficiency: {ntrig}/{nsim} = {100*ntrig/nsim:.1f}%")
    
    return ntrig, nsim


def preprocess_neutron_file(file_path, alpha, output_path):
    """
    Neutron scaling by beam power happens during analysis, not here.
    Just extract the raw energy/direction/mcke for later resampling.
    
    Parameters
    ----------
    file_path : str
        Path to neutron root file
    alpha : float
        Energy calibration constant
    output_path : Path
        Where to save the npz file
        
    Returns
    -------
    ntrig : int
        Number of triggered events
    nsim : int
        Number of simulated events
    """
    if not os.path.exists(file_path):
        print(f"  ERROR: File not found: {file_path}")
        return 0, 0
    
    print(f"  Processing neutron file: {file_path}")
    start_time = datetime.now()
    
    try:
        f = uproot.open(file_path)
        tree = f['output']
        
        # Load arrays
        triggerTime = tree["triggerTime"].array(library="np")
        x_quadfitter = tree['x_quadfitter'].array(library='np')
        hitPMTCharge_ak = tree["hitPMTCharge"].array(library="ak")
        u_fitdirection = tree["u_fitdirectioncenter_0p5_quad"].array(library="np")
        mcke = tree["mcke"].array(library="np")
        nsim = len(mcke)
        
        # Apply cuts
        nevents = min(len(triggerTime), len(x_quadfitter), len(u_fitdirection), len(mcke))
        triggerTime = triggerTime[:nevents]
        x_quadfitter = x_quadfitter[:nevents]
        hitPMTCharge_ak = hitPMTCharge_ak[:nevents]
        u_fitdirection = u_fitdirection[:nevents]
        mcke = mcke[:nevents]
        
        mask = (triggerTime > 0) & (x_quadfitter > -9999)
        
        # Compute energy (still need to sum jagged array for neutrons)
        print(f"    Computing total charge (this takes a moment for neutrons)...")
        energy = np.array([np.sum(charges) * alpha for charges in hitPMTCharge_ak[mask]])
        direction = u_fitdirection[mask]
        mcke_cut = mcke[mask]
        ntrig = len(energy)
        
        # Save with mcke for later beam power scaling
        np.savez_compressed(output_path,
                           energy=energy,
                           direction=direction,
                           mcke=mcke_cut,  # Keep this for spectrum resampling
                           ntrig=ntrig,
                           nsim=nsim)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        file_size_mb = output_path.stat().st_size / 1e6
        
        print(f"  ✓ Saved {ntrig:,} neutron events to {output_path}")
        print(f"    File size: {file_size_mb:.1f} MB")
        print(f"    Processing time: {elapsed:.1f} seconds")
        
        return ntrig, nsim
        
    except Exception as e:
        print(f"  ERROR: Failed to process neutron file: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0


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
        
        # Process neutron files
        print(f"\nProcessing neutron files:")
        for shielding, file_path in detector_config['neutrons'].items():
            if file_path is None:
                print(f"\n{shielding}: SKIPPED (file path not available)")
                continue
            
            print(f"\n{shielding}:")
            
            output_filename = f"{detector_name}_neutrons_{shielding}.npz"
            output_path = cfg.PREPROCESSED_DIR / output_filename
            
            # Check if already processed
            if output_path.exists():
                print(f"  Already processed: {output_path}")
                print(f"  Delete this file if you want to reprocess")
                continue
            
            ntrig, nsim = preprocess_neutron_file(file_path, alpha, output_path)
    
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
