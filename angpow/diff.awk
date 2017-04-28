
BEGIN { max=0. }
{ if(($2-$4)*($2-$4)>max){idx=$1;max=($2-$4)*($2-$4)} }
END{ print idx,sqrt(max) }
