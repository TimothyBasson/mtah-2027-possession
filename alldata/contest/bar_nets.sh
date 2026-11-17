# The all-match nets for BAR-ATH (fold none), both variants, two seeds; resumable.
J="sh alldata/contest/job.sh"

$J crN none --extra --brief 3 --focal 1 & $J crNs1 none --extra --brief 3 --focal 1 --seed 1 &
$J cN none --extra & $J cNs1 none --extra --seed 1 & wait
python3 alldata/learned/ens.py crNe crN crNs1; python3 alldata/learned/ens.py cNe cN cNs1
$J pcrN none --extra --brief 3 --focal 1 --pseudo crNe & $J pcrNs1 none --extra --brief 3 --focal 1 --pseudo crNe --seed 1 &
$J pcN none --extra --pseudo cNe & $J pcNs1 none --extra --pseudo cNe --seed 1 & wait
echo q4 done
