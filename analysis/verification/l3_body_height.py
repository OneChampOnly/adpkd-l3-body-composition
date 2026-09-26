"""Craniocaudal extent of every manual L3 mask (cross-check on the 20 mm rule)."""
import nibabel as nib, numpy as np, glob, os, csv
from pathlib import Path
ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[2]))
out=[]
for m in sorted(glob.glob(str(ROOT / 'dataset/*/l3_mask.nii.gz'))):
    cid=os.path.basename(os.path.dirname(m)); im=nib.load(m)
    a=np.asanyarray(im.dataobj)>0
    nz=np.flatnonzero(a.any(axis=(0,1)))
    zs=float(im.header.get_zooms()[2])
    out.append((cid, len(nz), (nz.max()-nz.min()+1)*zs, zs))
import statistics as st
h=[r[2] for r in out]
print('L3 수직 높이 (수동 마스크 craniocaudal extent), n=%d'%len(out))
print('  mean %.1f  SD %.1f  median %.1f  IQR %.1f-%.1f  min %.1f  max %.1f'%(
    st.mean(h),st.pstdev(h),st.median(h),np.percentile(h,25),np.percentile(h,75),min(h),max(h)))
with open(str(Path(__file__).resolve().parent / 'l3_height.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['case_id','n_slices','height_mm','native_spacing']); w.writerows(out)
