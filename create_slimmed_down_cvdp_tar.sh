#!/bin/bash
# This script will create a .tar.gz file containing the data and put it on a sensible web-location for downloading.
# Written by Chris Brierley and only to be used by Chris.

#first define a little function to say whether a netcdf file has required variables
# define a function to test whether the contents of the netcdf file is a regular lat,lon) file
function hasIODvars {
  hasIODvars_DIR=$1
  hasIODvars_filename=$2
  hasIODvars_iod_vars=`ncdump -h $hasIODvars_DIR/$hasIODvars_filename | grep float | grep iod | cut -d\( -f1 | cut -d\  -f2`
  if [[ $hasIODvars_iod_vars == *"iod_pr_regression_mon"* ]] && [[ $hasIODvars_iod_vars == *"iod_pattern_mon"* ]] && [[ $hasIODvars_iod_vars == *"iod_tas_regression_mon"* ]] && [[ $hasIODvars_iod_vars == *"iod_timeseries_mon"* ]]
  then
    return 1
  else
    return 0
  fi
}  

function hasTASvars {
  hasTASvars_DIR=$1
  hasTASvars_filename=$2
  hasTASvars_tas_vars=`ncdump -h $hasTASvars_DIR/$hasTASvars_filename | grep float | grep tas | cut -d\( -f1 | cut -d\  -f2`
  if [[ $hasTASvars_tas_vars == *"tas_spatialmean_ann"* ]]; then
    return 1
  else
    return 0
  fi
}  

function hasPRvars {
  hasPRvars_DIR=$1
  hasPRvars_filename=$2
  hasPRvars_pr_vars=`ncdump -h $hasPRvars_DIR/$hasPRvars_filename | grep float | grep pr | cut -d\( -f1 | cut -d\  -f2`
  if [[ $hasPRvars_pr_vars == *"pr_spatialmean_ann"* ]]; then
    return 1
  else
    return 0
  fi
}  

function hasSSTvars {
  hasSSTvars_DIR=$1
  hasSSTvars_filename=$2
  hasSSTvars_pr_vars=`ncdump -h $hasSSTvars_DIR/$hasSSTvars_filename | grep float | grep sst | cut -d\( -f1 | cut -d\  -f2`
  if [[ $hasPRvars_pr_vars == *"sst_spatialmean_ann"* ]]; then
    return 1
  else
    return 0
  fi
}  


CVDP_DATA_DIR=`pwd`"/data/full_files"
REPO_DATA_DIR=`pwd`"/data" #relative to here
plotting_vars="iod_pr_regression_mon,iod_pattern_mon,pr_spatialmean_ann,pr_spatialmean_djf,pr_spatialmean_jja,sst_spatialmean_ann,sst_spatialmean_djf,sst_spatialmean_jja,tas_spatialmean_ann,tas_spatialmean_djf,tas_spatialmean_jja,nino34,iod_timeseries_mon,tropical_indian_ocean,north_tropical_atlantic,east_eq_indian_ocean,monsoon_rain_SAS,monsoon_area_SAS,ipcc_SAS_lnd_pr"
IOD_vars="iod_pattern_mon,iod_pr_regression_mon,iod_tas_regression_mon,iod_timeseries_mon"
SST_vars="sst_spatialmean_ann,sst_spatialmean_djf,sst_spatialmean_jja,tropical_indian_ocean,east_eq_indian_ocean,nino34"
PR_vars="pr_spatialmean_ann,pr_spatialmean_djf,pr_spatialmean_jja,monsoon_rain_SAS,monsoon_area_SAS,ipcc_SAS_lnd_pr"
TAS_vars="tas_spatialmean_ann,tas_spatialmean_djf,tas_spatialmean_jja,ipcc_SAS_lnd_tas"

cd $CVDP_DATA_DIR
ncfiles=`ls *{piControl,midHolocene-cal-adj,lgm-cal-adj,lig127k-cal-adj,abrupt4xCO2}.cvdp_data.*-*.nc C20*nc`
echo $ncfiles
cd $REPO_DATA_DIR
for ncfile in $ncfiles
do
  echo working on $ncfile
  hasIODvars $CVDP_DATA_DIR $ncfile
  if [ $? == 1 ]; then
    ncks -O -v $IOD_vars $CVDP_DATA_DIR/$ncfile $ncfile
    ncks -A -v $SST_vars $CVDP_DATA_DIR/$ncfile $ncfile
    ncks -A -v $PR_vars $CVDP_DATA_DIR/$ncfile $ncfile
    ncks -A -v $TAS_vars $CVDP_DATA_DIR/$ncfile $ncfile
  fi 
done


#make a .tar.gz archive
#rm PMIP4_cvdp_SST.tar.gz
#tar -czf PMIP4_cvdp_SST.tar.gz *.cvdp_SST.*-*.nc
#cp PMIP4_cvdp_SST.tar.gz ~/public_html/PMIPVarData/data/PMIP4_cvdp_SST.tar.gz
