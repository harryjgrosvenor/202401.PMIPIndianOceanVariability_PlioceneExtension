;This script was used to create sufficient figures to populate a research publication
;It makes full use of the programs in cvdp_data.functions.ncl to plot figures solely from the output data of cvdp (without needing recomputation).
;The code it set up to look at the Indian Ocean Dipole.
; It makes 1 table (showing simulations used) and many figures. 

;load some functions (first 4 get loaded automatically on newer versions of NCL
load "cvdp_data.functions.ncl"

mode_name="iod";variable in the file
plot_regions=(/(/-35,35/),(/30,130/)/);lat and lons of the edge of the plots (/(/lonW,lonE/),(/latS,latN/)/)
index_region1=(/(/-5.,20./),(/-80.,0./)/);lat and lons of the edge of first index box (only used for scatter plot)
index_region2=(/(/-10.,5./),(/-60.,15./)/);lat and lons of the edge of second index box (only used for scatter plot, ignored if missing!)

expts=(/"piControl","midHolocene-cal-adj","lig127k-cal-adj","lgm-cal-adj","abrupt4xCO2"/)
expts_color=(/"grey24","black","darkgreen","dodgerblue4","yellowgreen","firebrick","orangered2"/)
gcms_all=(/"ACCESS-ESM1-5","AWI-ESM-1-1-LR","BCC-CSM1-1","CCSM4","CESM2","CNRM-CM5","CNRM-CM6-1","COSMOS-ASO","CSIRO-Mk3-6-0","CSIRO-Mk3L-1-2","EC-EARTH-2-2","EC-Earth3-LR","FGOALS-f3-L","FGOALS-g2","FGOALS-g3","FGOALS-s2","GISS-E2-1-G","GISS-E2-R","HadGEM2-CC","HadGEM2-ES","HadGEM3-GC31-LL","INM-CM4-8","IPSL-CM5A-LR","IPSL-CM6A-LR","KCM1-2-2","MIROC-ES2L","MIROC-ESM","MPI-ESM1-2-LR","MPI-ESM-P","MRI-CGCM3","MRI-ESM2-0","NESM3","NorESM1-F","NorESM2-LM","UofT-CCSM-4"/)
gcms_CMIP6=(/"ACCESS-ESM1-5","AWI-ESM-1-1-LR","CESM2","CNRM-CM6-1","EC-Earth3-LR","FGOALS-f3-L","FGOALS-g3","GISS-E2-1-G","HadGEM3-GC31-LL","INM-CM4-8","IPSL-CM6A-LR","MIROC-ES2L","MPI-ESM1-2-LR","MRI-ESM2-0","NESM3","NorESM2-LM"/)
gcms_CMIP6_subset=(/"ACCESS-ESM1-5","AWI-ESM-1-1-LR","CNRM-CM6-1","EC-Earth3-LR","FGOALS-f3-L","FGOALS-g3","GISS-E2-1-G","INM-CM4-8","IPSL-CM6A-LR","MIROC-ES2L","MPI-ESM1-2-LR","MRI-ESM2-0","NESM3","NorESM2-LM"/) ;ECS Screen v. likely from Hausfather et al (2022)

;some plot switches
PLOT_TAS_HIST=False;if True, plot the mean DJF & JJA temperature changes across the ensemble and their biases
TS_NOT_TAS=True ;if True use SST rather TAS in the mean state figure
PLOT_PR_HIST=False ;if True, plot the mean DJF & JJA precipiation changes across the ensemble and their biases
PLOT_HIST_MODE=False ;if True, plot the ATL3 and AMM patterns in the C20 Reanalysis and the Historical Simulations 
PLOT_TAS_ANOM=False ;if True, plot the mean DJF & JJA temperature changes across the ensemble and their biases
PLOT_PR_ANOM=False ;if True, plot the mean DJF & JJA precipiation changes across the ensemble and their biases
PLOT_PR_TELE_HIST=False ;if True, plot the precipitation patterns associated with the ATL3 and AMM indices in the C20 Reanalysis and the Historical Simulations 
PLOT_MIDH=False ;if True, for the midHolocene plot the ensemble mean change in ATL3, AMM, and SASD pattern and (with amplitude up top)
PLOT_LGM=False ;if True, for the LGM plot the ensemble mean change in ATL3, AMM, and SASD pattern and (with amplitude up top)
PLOT_FUTURE=False ;if True, for the 1pctCO2 run plot the ensemble mean change in ATL3, AMM, and SASD pattern and (with amplitude up top)
ADD_PRECIP_TO_PLOT=True ;if True, then adds the precipitation regression patterns to these plots
PLOT_TS=False ;if True, this spews out a series of timeseries (more for inspection than publication)
PRINT_TABLE=True ;if True, create a table (in LaTeX format) showing the simulations used 
GCM_DICTIONARY=False; if True, will not print model names, but alphabetic look up from table
BARCHART=False ;if True, plot some barcharts showing how the amplitude (i.e. stddev of timeseries) changes for ATL3 and AMM (warning not relevant for Principal Component based timeseries as these are normalised)
MAKE_INDIVIDUAL_PLOTS=False; if True will actually create the plots. ;Sets which experiment the barchart is created for (as index of expts array above)
PLOT_MODE_SCATTER=False ;if True, create a scatterplot of the change in MODE amplitude vs the change in mean SST.
SCATTER_PCT_CHANGES=True ;if True, the scatterplots present the changes as percentages.
VERBOSE=False ;if True, provides a modicum of more information about whats goign on in the plots - used for debugging mainly.
OUTPUT_TYPE="pdf"; either "png" or "pdf"
USE_CMIP6_SUBSET_GCMS=False; if True then only use those GCMs that have done all the runs
ONLY_CMIP6_GCMS=False; if True then only use those GCMs that have done all the runs
INC_PREIND_wHIST=True; adds the preindustrial to all the plots including the Historical 
OVERLAY_PREIND=True; Adds contour line overlays of the piControl values (to help eye decide on locations of changes).

;;;;;;;;;;;
;;Process the GCM choices
;;;;;

paper_str="plots/initial_iod_plots_"
data_dir="/home/p2f-v/public_html/PMIPVarData/cvdp_files"
if GCM_DICTIONARY then
  gcm_dict=createGCMsNameDictionary(expts,mode_name+"_timeseries_mon")
end if
plot_letters=(/(/"a","b"/),(/"c","d"/),(/"e","f"/),(/"g","h"/),(/"i","j"/),(/"k","l"/)/)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Plot the historical ensemble mean ;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
if PLOT_HIST_MODE then
  ;Find some standard deviation for plot labelling
  RightStrings=new(3,string)
  fname_obs=find_files_wVar("C20-Reanalysis",mode_name+"_timeseries_mon")
  RightStrings(0)=sprintf("%5.2f",stat_ts_var(fname_obs,mode_name+"_timeseries_mon","stddev",False))
  fnames_hist=find_files_wVar(expts(1),mode_name+"_timeseries_mon")
  stddevs=new(dimsizes(fnames_hist),float)
  do mod_i=0,dimsizes(fnames_hist)-1
    stddevs(mod_i)=stat_ts_var(fnames_hist(mod_i),mode_name+"_timeseries_mon","stddev",False)
  end do
  RightStrings(1)=sprintf("%5.2f",avg(stddevs))
  delete(stddevs)
  if INC_PREIND_wHIST then
    fnames_preind=find_files_wVar(expts(0),mode_name+"_timeseries_mon")
    stddevs=new(dimsizes(fnames_preind),float)
    do mod_i=0,dimsizes(fnames_preind)-1
      stddevs(mod_i)=stat_ts_var(fnames_preind(mod_i),mode_name+"_timeseries_mon","stddev",False)
    end do
    RightStrings(2)=sprintf("%5.2f",avg(stddevs))
  end if
  
  opt=True
  ;opt@gsnRightString="~F33~s~F21~="+RightStrings+"~F35~J~F21~C" 
  opt@gsnRightString=RightStrings+"~F35~J~F21~C"
  opt@gsnLeftString=(/"a","b","c"/)
  opt@lbLabelBarOn=True
  opt@mpProjection="CylindricalEqualArea"
  opt@gsnStringFontHeightF=0.03
  opt@lbLabelFontHeightF=0.018
  opt@mpLimitMode="LatLon"
  opt@mpMinLonF=plot_regions(1,0)
  opt@mpMaxLonF=plot_regions(1,1)
  opt@mpMinLatF=plot_regions(0,0)
  opt@mpMaxLatF=plot_regions(0,1)
  opt@lbLabelBarOn=False
  opt@cnMinLevelValF = -0.8
  opt@cnMaxLevelValF = 0.8
  opt@cnLevelSpacingF = 0.1
  opt@cnLevelSelectionMode = "ManualLevels"
  opt_pan=True
  opt_pan@lbTitleString="Linear Regression on Index (~F35~J~F21~C/~F35~J~F21~C)"
  opt_pan@lbTitleFontHeightF=0.02
  opt_pan@lbLabelFontHeightF=0.016
  opt_pan@gsnPanelLabelBar=True
  opt_pan@lbTitlePosition="Bottom"
  plotname=paper_str+"hist_mode"
  plotname@filetype=OUTPUT_TYPE
  plotname@VERBOSE=VERBOSE
  if INC_PREIND_wHIST then
    plotCVDPcomparisonMaps((/"C20-Reanalysis",expts(1),expts(0)/),mode_name+"_pattern_mon",\
          plotname,opt,opt_pan)
  else
    plotCVDPcomparisonMaps((/"C20-Reanalysis",expts(1)/),mode_name+"_pattern_mon",\
          plotname,opt,opt_pan)
  end if
  delete([/RightStrings,fname_obs,fnames_hist,opt,opt_pan/])
end if;PLOT_HIST_MODE_AMM

if PLOT_TAS_HIST then
  ;Rather than immediately sending the plot to a file, this plot first opens a wks and panels within it. 
  if TS_NOT_TAS then
    wks=gsn_open_wks(OUTPUT_TYPE,paper_str+"ts_hist")
    djf_field="sst_spatialmean_djf"
    jja_field="sst_spatialmean_jja"
  else
    wks=gsn_open_wks(OUTPUT_TYPE,paper_str+"tas_hist")
    djf_field="tas_spatialmean_djf"
    jja_field="tas_spatialmean_jja"
  end if
  wks@VERBOSE=VERBOSE
  opt=True
  opt@cnLevelSelectionMode = "ManualLevels"   
  if TS_NOT_TAS then
    opt@cnMinLevelValF       = 15
    opt@cnMaxLevelValF       = 30
  else
    opt@cnMinLevelValF       = 10
    opt@cnMaxLevelValF       = 35
  end if
  opt@cnLevelSpacingF = 1
  opt@cnFillPalette="WhiteBlueGreenYellowRed"
  opt@mpProjection="CylindricalEqualArea"
  opt@gsnStringFontHeightF=0.03
  opt@lbLabelFontHeightF=0.018
  opt@mpLimitMode="LatLon"
  opt@mpMinLonF=plot_regions(1,0)
  opt@mpMaxLonF=plot_regions(1,1)
  opt@mpMinLatF=plot_regions(0,0)
  opt@mpMaxLatF=plot_regions(0,1)
  opt@gsnLeftString=(/"a","b"/)
  opt@tiMainString=""
  opt@gsnCenterString=""
  opt@lbLabelBarOn=False
  opt_pan=True
  opt_pan@lbTitleOn=False
  opt_pan@lbOrientation = "vertical"
  opt_pan@gsnPanelLabelBar=True
  if INC_PREIND_wHIST then
    opt_pan@gsnPanelBottom=0.66
  else
    opt_pan@gsnPanelBottom=0.5
  end if
  opt_pan@gsnFrame=False
  opt_pan@lbLabelFontHeightF=0.016
  if TS_NOT_TAS then
    opt@gsnRightString="HadISST "+(/"DJF","JJA"/)
    plotCVDPcomparisonMaps("HadISST",(/djf_field,jja_field/),wks,opt,opt_pan)
  else
    opt@gsnRightString="C20-Reanalysis"+(/"DJF","JJA"/)
    plotCVDPcomparisonMaps("C20-Reanalysis",(/djf_field,jja_field/),wks,opt,opt_pan)
  end if
  opt@cnFillPalette="CBR_coldhot"
  opt@cnLevelSelectionMode = "ExplicitLevels"   
  opt@cnLevels=(/-5.,-3.,-2.,-1.,-0.5,0.5,1.,2.,3.,5./)
  opt_pan@lbTitleString="Temperature Difference (oC)"
  opt@gsnLeftString=(/"c","d"/)
  opt@gsnRightString="historical "+(/"DJF","JJA"/)
  opt@CONSISTENCY=True ;Turn on Stippling
  if INC_PREIND_wHIST then
    opt_pan@gsnPanelBottom=0.33
    opt_pan@gsnPanelTop=0.66
  else
    opt_pan@gsnPanelBottom=0.0
    opt_pan@gsnPanelTop=0.5
  end if
  opt_pan@dims=(/1,2/)
  if TS_NOT_TAS then
    plotDiffEnsMnMaps(expts(1),"HadISST",(/djf_field,jja_field/),wks,opt,opt_pan)
  else
    plotDiffEnsMnMaps(expts(1),"C20-Reanalysis",(/djf_field,jja_field/),wks,opt,opt_pan)
  end if
  if INC_PREIND_wHIST then
    opt@cnFillPalette="CBR_coldhot"
    opt@cnLevelSelectionMode = "ExplicitLevels"   
    opt@cnLevels=(/-5.,-3.,-2.,-1.,-0.5,0.5,1.,2.,3.,5./)
    opt_pan@lbTitleString="Temperature Difference (oC)"
    opt@gsnLeftString=(/"c","d"/)
    opt@gsnRightString="piControl "+(/"DJF","JJA"/)
    opt@CONSISTENCY=True ;Turn on Stippling
    opt_pan@gsnPanelBottom=0.0
    opt_pan@gsnPanelTop=0.33
    opt_pan@dims=(/1,2/)
    if TS_NOT_TAS then
      plotDiffEnsMnMaps(expts(0),"HadISST",(/djf_field,jja_field/),wks,opt,opt_pan)
    else
      plotDiffEnsMnMaps(expts(0),"C20-Reanalysis",(/djf_field,jja_field/),wks,opt,opt_pan)
    end if
  end if
  frame(wks)
  delete(wks)
  delete([/opt,opt_pan/])
end if;PLOT_TAS_HIST

if PLOT_TAS_ANOM then
  ;Rather than immediately sending the plot to a file, this plot first opens a wks and panels within it. 
  if TS_NOT_TAS then
    wks=gsn_open_wks(OUTPUT_TYPE,paper_str+"ts_mean")
    djf_field="sst_spatialmean_djf"
    jja_field="sst_spatialmean_jja"
  else
    wks=gsn_open_wks(OUTPUT_TYPE,paper_str+"tas_mean")
    djf_field="tas_spatialmean_djf"
    jja_field="tas_spatialmean_jja"
  end if
  wks@VERBOSE=VERBOSE
  opt=True
  opt@cnFillPalette="CBR_coldhot"
  opt@cnLevelSelectionMode = "ExplicitLevels"   
  opt@cnLevels=(/-5.,-3.,-2.,-1.,-0.5,0.5,1.,2.,3.,5./)
  opt@mpProjection="CylindricalEqualArea"
  opt@mpLimitMode="LatLon"
  opt@mpMinLonF=plot_regions(1,0)
  opt@mpMaxLonF=plot_regions(1,1)
  opt@mpMinLatF=plot_regions(0,0)
  opt@mpMaxLatF=plot_regions(0,1)
  opt@gsnStringFontHeightF=0.03
  opt@lbLabelFontHeightF=0.018
  opt@tiMainString=""
  opt@gsnCenterString=""
  opt@lbLabelBarOn=False
  opt@CONSISTENCY=True ;Turn on Stippling
  opt@OVERLAY_CONTROL=OVERLAY_PREIND
  opt@OVERLAY_CONTROL_MANUALLEVS=(/20,32,1/);ie 0:12:2 mm/day
  opt_pan=True
  opt_pan@lbTitleOn=False
  opt_pan@lbTitleString="Temperature Difference (oC)"
  opt_pan@lbOrientation = "vertical"
  opt_pan@gsnPanelLabelBar=True
  opt_pan@gsnPanelBottom=0.66
  opt_pan@lbLabelFontHeightF=0.016
  opt_pan@gsnFrame=False
  opt_pan@dims=(/1,2/)

  n_expts=dimsizes(expts)-2
  do i=0,n_expts-1
    opt@gsnLeftString=plot_letters(i,:)
    opt@gsnRightString=expts(i+2)+(/" DJF"," JJA"/)
    opt_pan@gsnPanelTop=1-(tofloat(i)/n_expts)
    opt_pan@gsnPanelBottom=1-(tofloat(i+1)/n_expts)
    plotDiffEnsMnMaps(expts(i+2),expts(0),(/"tas_spatialmean_djf","tas_spatialmean_jja"/),wks,opt,opt_pan)
  end do

  frame(wks)
  delete(wks)
  delete([/opt,opt_pan/])
end if;PLOT_TAS_ANOM

if PLOT_PR_HIST then
  wks=gsn_open_wks(OUTPUT_TYPE,paper_str+"pr_hist")
  wks@VERBOSE=VERBOSE
  opt=True
  opt@cnLevelSelectionMode = "ManualLevels"   
  opt@cnMinLevelValF       = 0
  opt@cnMaxLevelValF       = 10
  opt@cnLevelSpacingF = 1.
  opt@cnFillPalette="CBR_wet"
  opt@mpProjection="CylindricalEqualArea"
  opt@mpLimitMode="LatLon"
  opt@mpMinLonF=plot_regions(1,0)
  opt@mpMaxLonF=plot_regions(1,1)
  opt@mpMinLatF=plot_regions(0,0)
  opt@mpMaxLatF=plot_regions(0,1)
  opt@gsnLeftString=(/"a","b"/)
  opt@gsnRightString="GPCP "+(/"DJF","JJA"/)
  opt@gsnStringFontHeightF=0.03
  opt@tiMainString=""
  opt@gsnCenterString=""
  opt@lbLabelBarOn=False
  opt_pan=True
  opt_pan@lbTitleOn=False
  opt_pan@lbOrientation = "vertical"
  opt_pan@gsnPanelLabelBar=True
  opt_pan@lbLabelFontHeightF=0.016
  if INC_PREIND_wHIST then
    opt_pan@gsnPanelBottom=0.66
  else
    opt_pan@gsnPanelBottom=0.5
  end if
  opt_pan@gsnFrame=False
  plotCVDPcomparisonMaps("GPCP",(/"pr_spatialmean_djf","pr_spatialmean_jja"/),wks,opt,opt_pan)
  opt@cnFillPalette="CBR_drywet"
  opt@cnLevelSelectionMode = "ExplicitLevels"   
  opt@cnLevels=(/-5.,-2.,-1.,-0.5,-0.1,0.1,0.5,1.,2.,5./)
  opt_pan@lbTitleString="Precip. Difference (mm/day)"
  opt@gsnLeftString=(/"c","d"/)
  opt@gsnRightString="historical "+(/"DJF","JJA"/)
  opt@CONSISTENCY=True ;Turn on Stippling
  if INC_PREIND_wHIST then
    opt_pan@gsnPanelTop=0.66
    opt_pan@gsnPanelBottom=0.33
  else
    opt_pan@gsnPanelBottom=0.0
    opt_pan@gsnPanelTop=0.5
  end if
  opt_pan@dims=(/1,2/)
  plotDiffEnsMnMaps(expts(1),"GPCP",(/"pr_spatialmean_djf","pr_spatialmean_jja"/),wks,opt,opt_pan)
  if INC_PREIND_wHIST then
    opt@cnFillPalette="CBR_drywet"
    opt@cnLevelSelectionMode = "ExplicitLevels"   
    opt@cnLevels=(/-5.,-2.,-1.,-0.5,-0.1,0.1,0.5,1.,2.,5./)
    opt_pan@lbTitleString="Precip. Difference (mm/day)"
    opt@gsnLeftString=(/"e","f"/)
    opt@gsnRightString="piControl "+(/"DJF","JJA"/)
    opt@CONSISTENCY=True ;Turn on Stippling
    opt_pan@gsnPanelBottom=0.0
    opt_pan@gsnPanelTop=0.33
    opt_pan@dims=(/1,2/)
    plotDiffEnsMnMaps(expts(0),"GPCP",(/"pr_spatialmean_djf","pr_spatialmean_jja"/),wks,opt,opt_pan)
  end if
  frame(wks)
  delete(wks)
  delete([/opt,opt_pan/])
end if;PLOT_PR_HIST

if PLOT_PR_ANOM then
  wks=gsn_open_wks(OUTPUT_TYPE,paper_str+"pr_mean")
  wks@VERBOSE=VERBOSE
  opt=True
  opt@mpProjection="CylindricalEqualArea"
  opt@mpLimitMode="LatLon"
  opt@mpMinLonF=plot_regions(1,0)
  opt@mpMaxLonF=plot_regions(1,1)
  opt@mpMinLatF=plot_regions(0,0)
  opt@mpMaxLatF=plot_regions(0,1)
  opt@tiMainString=""
  opt@gsnCenterString=""
  opt@lbLabelBarOn=False
  opt@cnFillPalette="CBR_drywet"
  opt@cnLevelSelectionMode = "ExplicitLevels"   
  opt@cnLevels=(/-5.,-2.,-1.,-0.5,-0.1,0.1,0.5,1.,2.,5./)
  opt@gsnStringFontHeightF=0.03
  opt@CONSISTENCY=True ;Turn on stippling
  opt@OVERLAY_CONTROL=OVERLAY_PREIND
  opt@OVERLAY_CONTROL_MANUALLEVS=(/0,12,2/);ie 0:12:2 mm/day
  opt_pan=True
  opt_pan@lbTitleOn=False
  opt_pan@lbOrientation = "vertical"
  opt_pan@gsnPanelLabelBar=True
  opt_pan@gsnFrame=False
  opt_pan@lbTitleString="Precip. Difference (mm/day)"
  opt_pan@lbLabelFontHeightF=0.016
  opt_pan@dims=(/1,2/)
  
  
  n_expts=dimsizes(expts)-2
  do i=0,n_expts-1
    opt@gsnLeftString=plot_letters(i,:)
    opt@gsnRightString=expts(i+2)+(/" DJF"," JJA"/)
    opt_pan@gsnPanelTop=1-(tofloat(i)/n_expts)
    opt_pan@gsnPanelBottom=1-(tofloat(i+1)/n_expts)
    plotDiffEnsMnMaps(expts(i+2),expts(0),(/"pr_spatialmean_djf","pr_spatialmean_jja"/),wks,opt,opt_pan)
  end do

  frame(wks)
  delete(wks)
  delete([/opt,opt_pan/])
end if;PLOT_PR_ANOM

if PLOT_PR_TELE_HIST then
  if VERBOSE then
    print("Plotting precipitation regressions")
  end if
  plotname=paper_str+"piControl_precip_regress"
  plotname@filetype=OUTPUT_TYPE
  plotname@VERBOSE=VERBOSE
  opt=True
  opt@mpProjection="CylindricalEqualArea"
  opt@mpLimitMode="LatLon"
  opt@mpMinLonF=plot_regions(1,0)
  opt@mpMaxLonF=plot_regions(1,1)
  opt@mpMinLatF=plot_regions(0,0)
  opt@mpMaxLatF=plot_regions(0,1)
  opt@tiMainString=""
  opt@gsnCenterString=""
  opt@lbLabelBarOn=False
  opt@gsnStringFontHeightF=0.04
  opt@gsnLeftString=(/"a","b","c"/)
  opt@gsnRightString=(/"Reanalysis","historical","piControl"/)
  opt@cnFillPalette="CBR_drywet"
  opt@cnLevelSelectionMode = "ExplicitLevels"   
  opt@cnLevels=(/-2.,-1.,-0.5,-0.2,-0.1,0.1,0.2,0.5,1.,2./)
  opt_pan=True
  opt_pan@lbTitleOn=True
  opt_pan@gsnPanelLabelBar=True
  opt_pan@lbTitleString="Precip. regression (mm/day/~F35~J~F21~C)"
  opt_pan@lbLabelFontHeightF=0.016
  plotCVDPcomparisonMaps((/"C20-Reanalysis",expts(0),expts(1)/),mode_name+"_pr_regression_mon",plotname,opt,opt_pan)
  delete(opt)
end if;PLOT_PR_TELE_HIST

if any((/PLOT_MIDH,PLOT_LGM,PLOT_FUTURE/)) then
  plots_to_make=ind((/PLOT_MIDH,PLOT_LGM,PLOT_FUTURE/))
  do i=0,dimsizes(plots_to_make)-1
    opt=True
    opt@CONSISTENCY=True
    opt@OVERLAY_CONTROL=OVERLAY_PREIND
    opt@OVERLAY_CONTROL_MANUALLEVS=(/-1.,1.,0.2/)
    opt@mpProjection="CylindricalEqualArea"
    opt@mpLimitMode="LatLon"
    opt@lbLabelBarOn=False
    opt@mpMinLonF=plot_regions(1,0)
    opt@mpMaxLonF=plot_regions(1,1)
    opt@mpMinLatF=plot_regions(0,0)
    opt@mpMaxLatF=plot_regions(0,1)
    opt@gsnCenterString=""
    opt@lbTitleFontHeightF=0.016
    opt@lbLabelFontHeightF=0.016
    optpan=True
    optpan@lbTitleFontHeightF=0.01
    optpan@lbLabelFontHeightF=0.016
    if ADD_PRECIP_TO_PLOT then
      wks=gsn_open_wks(OUTPUT_TYPE,paper_str+"pr_reg_"+expts(plots_to_make(i)+2))
      wks@VERBOSE=VERBOSE
      opt@gsnStringFontHeightF=0.04
      opt@gsnLeftString=(/"a"/)
      opt@gsnCenterString=""
      opt@tiMainString=""
      optpan@gsnPanelRight=0.5
      optpan@gsnFrame=False
      optpan@gsnPanelLabelBar=True
      optpan@lbTitleString="~F35~J~F21~C/~F35~J~F21~C"
      opt@cnLevelSelectionMode = "ExplicitLevels"   
      opt@cnLevels=2*(/-0.1,-0.08,-0.06,-0.04,-.02,0.02,.04,0.06,0.08,.1/)
      plotDiffEnsMnMaps(expts(plots_to_make(i)+2),expts(0),mode_name+"_pattern_mon",wks,opt,optpan)
      opt@gsnLeftString=(/"b"/)
      opt@cnFillPalette="CBR_drywet"
      opt@cnLevelSelectionMode = "ExplicitLevels"   
      delete(opt@cnLevels)
      opt@cnLevels=(/-0.5,-0.3,-0.1,-0.05,0.05,0.1,0.3,0.5/)
      optpan@gsnFrame=True
      optpan@gsnPanelLabelBar=True
      optpan@lbTitleString="mm/day/~F35~J~F21~C"
      optpan@lbLabelFontHeightF=0.016
      optpan@gsnPanelRight=1.0
      optpan@gsnPanelLeft=0.5      
      plotDiffEnsMnMaps(expts(plots_to_make(i)+2),expts(0),mode_name+"_pr_regression_mon",wks,opt,optpan)
      delete([/wks,opt,optpan/])
    else
      plotname=paper_str+expts(plots_to_make(i)+2)
      plotname@filetype=OUTPUT_TYPE
      plotname@VERBOSE=VERBOSE
      opt@gsnLeftString=(/" "/)
      plotDiffEnsMnMaps(expts(plots_to_make(i)+2),expts(0),mode_name+"_pattern_mon",plotname,opt,optpan)
    end if
  end do
end if;PLOT_MIDH,PLOT_LGM,PLOT_FUTURE

if PLOT_TS then
  ts_opt=True
  ts_opt@RUN_STDDEV=True
  opt_res=True
  opt_res@tiYAxisString=mode_name+" stddev"
  opt_res@tiXAxisString=""
  opt_res@tiMainString="C20-Reanalysis"
  plotEnsTimeseries("C20-Reanalysis",mode_name+"_timeseries_mon",ts_opt,paper_str+"C20-Reanalysis_runstddev",opt_res,False)
  opt_res@tiMainString="Historical Simulations"
  plotEnsTimeseries("historical",mode_name+"_timeseries_mon",ts_opt,paper_str+"historical_runstddev",opt_res,False)
  opt_res@tiMainString="1%CO2 Simulations"
  plotEnsTimeseries("1pctCO2",mode_name+"_timeseries_mon",ts_opt,paper_str+"1pctCO2_runstddev",opt_res,False)
  delete(ts_opt@RUN_STDDEV)
  opt_res@tiYAxisString=mode_name
  opt_res@tiMainString="past1000"
  plotEnsTimeseries("past1000",mode_name+"_timeseries_mon",ts_opt,paper_str+"past100_runstddev",opt_res,False)
end if;PLOT_TS

if PRINT_TABLE then
  if GCM_DICTIONARY then
    createTableGCMsExptsYears(expts,mode_name+"_timeseries_mon",gcm_dict,"tex",paper_str+mode_name+"_table.tex")
  else
    createTableGCMsExptsYears(expts,mode_name+"_timeseries_mon",False,"tex",paper_str+mode_name+"_table.tex")
  end if
end if ;PRINT_TABLE

do expt_i=0,dimsizes(expts)-3
  this_expt=expts(expt_i+2)
  fnames_both=find_pair_files_wVar(this_expt,"piControl",mode_name+"_timeseries_mon")
  if VERBOSE then
    print(" ")
    print("expt, gcm_name, "+mode_name+"_stddev, control_"+mode_name+"_stddev, percentage change") 
  end if
  n_models=dimsizes(fnames_both(:,0))
  mode_sds=new((/n_models,3/),float)
  gcm_names=new(n_models,string)
  ts_opt=True
  ts_opt@NO_EXIT_ON_ERROR=True
  if any(this_expt.eq.(/"1pctCO2","abrupt4xCO2","rcp26","rcp85"/)) then
    ts_opt@subset=True
    ts_opt@subset_syear=-50
    ts_opt@subset_eyear=0
    ts_opt@renorm_climo=True
    ts_opt@renorm_climo_syear=-50
    ts_opt@renorm_climo_eyear=0
    ts_opt@DETREND=True 
  end if

  do i=0,n_models-1
    a_nopath=str_get_field(fnames_both(i,0),str_fields_count(fnames_both(i,0),"/"),"/")
    gcm_names(i)=str_get_field(str_sub_str(a_nopath,"_"+this_expt,":"),1,":")
    mode_sds(i,0)=(/stat_ts_var(fnames_both(i,0),mode_name+"_timeseries_mon","stddev",ts_opt)/)
    mode_sds(i,1)=(/stat_ts_var(fnames_both(i,1),mode_name+"_timeseries_mon","stddev",False)/)
    if VERBOSE then
      print(this_expt+", "+gcm_names(i)+", "+mode_sds(i,0)+", "+mode_sds(i,1)+", "+\
      (mode_sds(i,0)-mode_sds(i,1))/mode_sds(i,1)*100.+"%") 
    end if
  end do
  mode_sds(:,2)=(mode_sds(:,0)-mode_sds(:,1))/mode_sds(:,1)*100.
  print("Ensemble mean change in "+this_expt+" = "+avg(mode_sds(:,2))+"%") 
  
  if MAKE_INDIVIDUAL_PLOTS then
    plot_res = True
    plot_res@trYMinF = 0.
    plot_res@vpWidthF = 0.9
    plot_res@vpHeightF = 0.33
    plot_res@gsnDraw = False
    plot_res@gsnFrame=False
    plot_res@gsnXYBarChart = True
    plot_res@tiMainString = " "
    plot_res@gsnRightString = ""
    plot_res@tiXAxisString = " "
    plot_res@tmYMajorGrid = False
    plot_res@tmXTOn = False
    plot_res@tmXBMode = "Explicit"
    plot_res@tmXBValues = ispan(1,n_models,1)
    if GCM_DICTIONARY then
      gcm_letters=translateGCMsNameDictionary(gcm_names,gcm_dict,False)
      plot_res@tmXBLabels = gcm_letters
      plot_res@tmXBLabelFontHeightF = 0.018
      plot_res@tmEqualizeXYSizes=True
    else
      plot_res@tmXBLabels = gcm_names
      plot_res@tmXBLabelFontHeightF = 0.012
    end if
    plot_res@tmXBLabelDirection="Down"
    plot_res@trXMinF = 0.5
    plot_res@trXMaxF = n_models+0.5
    plot_res@trYMinF = 0.0
    plot_res@trYMaxF = 1.5
    plot_res@gsnYRefLine=0.
    plot_res@gsnXYBarChartBarWidth = 0.4
    left_locations=fspan(0.5,n_models-0.5,n_models)+0.3
    right_locations=left_locations+0.4
  
    wks = gsn_open_wks(OUTPUT_TYPE,paper_str+this_expt+"_amplitude_change")
    plot_res@tiYAxisString = "nino34"+" ~F33~s~F21~ (~F35~J~F21~C)"
    plot_res@gsnXYBarChartColors=expts_color(expt_i+2)
    mode_baseplot = gsn_csm_xy(wks,left_locations,mode_sds(:,0),plot_res)
    plot_res@gsnXYBarChartColors=expts_color(0)
    mode_oplot = gsn_csm_xy(wks,right_locations,mode_sds(:,1),plot_res)
    overlay(mode_baseplot,mode_oplot)

    plot_res@tiYAxisString = "Change in ~F33~s~F21~ (%)"
    plot_res@gsnXYBarChartColors="orange2"
    plot_res@trYMaxF = 50.
    plot_res@trYMinF = -50.
    diff_baseplot = gsn_csm_xy(wks,left_locations,mode_sds(:,2),plot_res)

    resP=True
    resP@gsnMaximize=True
    if .not.GCM_DICTIONARY then
      resP@gsnPanelYWhiteSpacePercent = 20.0
    end if
    gsn_panel(wks,(/mode_baseplot,diff_baseplot/),(/2,1/),resP)
    delete([/wks,resP,plot_res,mode_baseplot,diff_baseplot,ts_opt/])
    delete([/mode_oplot,left_locations,right_locations/])
  end if
  delete([/fnames_both,n_models,mode_sds,gcm_names/])
end do ;BARCHART

if PLOT_MODE_SCATTER then
  fnames_pictl=find_files_wVar("piControl",mode_name+"_timeseries_mon")
  num_pictl=dimsizes(fnames_pictl)
  d_stddev=new(num_pictl*5,float);array to hold variability change
  d_index=new(num_pictl*5,float);array to hold mean state change
  ts_opt=True
  mean_field="sst_spatialmean_ann"
  ts_1pctCO2=ts_opt
  ts_1pctCO2@renorm_climo=True
  ts_1pctCO2@subset=True
  ts_1pctCO2@subset_syear=-39
  ts_1pctCO2@subset_eyear=0
  ts_1pctCO2@renorm_climo_syear=-39
  ts_1pctCO2@renorm_climo_eyear=0
  
  ;collect all the data
  do expt_i=0,4
    if expts(expt_i+2).eq."1pctCO2" then
      ts_a=ts_1pctCO2
    else
      ts_a=ts_opt
    end if
    fnames_both=find_pair_files_wVar(expts(expt_i+2),"piControl",mode_name+"_timeseries_mon")
    do gcm_i=0,dimsizes(fnames_both(:,0))-1
      sd_a=stat_ts_var(fnames_both(gcm_i,0),mode_name+"_timeseries_mon","stddev",ts_a)
      sd_b=stat_ts_var(fnames_both(gcm_i,1),mode_name+"_timeseries_mon","stddev",ts_opt)
      region1_a=extract_latlon_areastat(fnames_both(gcm_i,0),mean_field,index_region1(0,:),index_region1(1,:),"mean")
      region1_b=extract_latlon_areastat(fnames_both(gcm_i,1),mean_field,index_region1(0,:),index_region1(1,:),"mean")
      if all(ismissing(index_region2)) then
        if SCATTER_PCT_CHANGES then
          d_stddev(expt_i*num_pictl+gcm_i)=100.*(sd_a-sd_b)/sd_b
          d_index(expt_i*num_pictl+gcm_i)=100.*(region1_a-region1_b)/(region1_b)
        else
          d_stddev(expt_i*num_pictl+gcm_i)=sd_a-sd_b
          d_index(expt_i*num_pictl+gcm_i)=region1_a-region1_b
        end if
        delete([/sd_a,sd_b,region1_a,region1_b/])
      else
        region2_a=extract_latlon_areastat(fnames_both(gcm_i,0),mean_field,index_region2(0,:),index_region2(1,:),"mean")
        region2_b=extract_latlon_areastat(fnames_both(gcm_i,1),mean_field,index_region2(0,:),index_region2(1,:),"mean")
        if SCATTER_PCT_CHANGES then
          d_stddev(expt_i*num_pictl+gcm_i)=100.*(sd_a-sd_b)/sd_b
          d_index(expt_i*num_pictl+gcm_i)=100.*((region1_a-region2_a)-(region1_b-region2_b))/(region1_b-region2_b)
        else
          d_stddev(expt_i*num_pictl+gcm_i)=sd_a-sd_b
          d_index(expt_i*num_pictl+gcm_i)=(region1_a-region2_a)-(region1_b-region2_b)
        end if
        delete([/sd_a,sd_b,region1_a,region2_a,region1_b,region2_b/])
      end if
    end do
    delete(fnames_both)
  end do

  ;create the scatter plot
  wks = gsn_open_wks(OUTPUT_TYPE,paper_str+mode_name+"_scatterplot")
  res                   = True                     ; plot mods desired
  res@gsnDraw = False
  res@gsnFrame = False                     ; plot mods desired
  res@tiMainString      = " "           ; add title
  res@xyMarkLineModes   = "Markers"                ; choose which have markers
  res@xyMarkers         =  16                      ; choose type of marker  
  res@xyMonoMarkerColor = True
  res@xyMarkerSizeF     = 0.01                     ; Marker size (default 0.01)
  res@gsnYRefLine=0.0
  res@gsnXRefLine=0.0
  res@tmLabelAutoStride = True                     ; nice tick mark labels
  if SCATTER_PCT_CHANGES then
    res@tiYAxisString   = mode_name+" Amplitude Change (%)"
    res@tiXAxisString   = mode_name+" Mean Change (%)"
  else
    res@tiYAxisString   = mode_name+" Amplitude Change"
    res@tiXAxisString   = mode_name+" Mean Change"
  end if
  ;res@trYMinF=min(d_stddev)
  ;res@trXMinF=min(d_index)
  ;res@trYMaxF=max(d_stddev)
  ;res@trXMaxF=max(d_index)
  res@trYMinF=-45.
  res@trYMaxF=75.
  res@trXMinF=-125.
  res@trXMaxF=125.
  
  res@xyMarkerColor =  expts_color(2) 
  plot  = gsn_csm_xy (wks,d_index(0:num_pictl-1),d_stddev(0:num_pictl-1),res) ; create plot
  res@xyMarkerColor =  expts_color(3) 
  oplot1  = gsn_csm_xy (wks,d_index(num_pictl:2*num_pictl-1),d_stddev(num_pictl:2*num_pictl-1),res)
  res@xyMarkerColor =  expts_color(4) 
  oplot2  = gsn_csm_xy (wks,d_index(2*num_pictl:3*num_pictl-1),d_stddev(2*num_pictl:3*num_pictl-1),res)
  res@xyMarkerColor =  expts_color(5) 
  oplot3  = gsn_csm_xy (wks,d_index(3*num_pictl:4*num_pictl-1),d_stddev(3*num_pictl:4*num_pictl-1),res)
  ;res@xyMarkerColor =  expts_color(6) 
  ;oplot4  = gsn_csm_xy (wks,d_index(4*num_pictl:5*num_pictl-1),d_stddev(4*num_pictl:5*num_pictl-1),res)
  overlay(plot,oplot1)
  overlay(plot,oplot2)
  overlay(plot,oplot3)
  ;overlay(plot,oplot4)
  draw(plot)
  frame(wks)
  delete(wks)
end if ;PLOT_MODE_SCATTER
