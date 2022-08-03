export CONDA_BUILD_SYSROOT=/

rm -rf class_public
git clone --depth=1000 https://github.com/lesgourg/class_public.git
cd class_public
git checkout v2.7.2

echo "----------------------------------"
env
echo "----------------------------------"

sed -i.bak -e 's/^CC/#CC/g' Makefile
sed -i.bak -e 's/^OPTFLAG =/OPTFLAG = ${CFLAGS} ${LDFLAGS}/g' Makefile
sed -i.bak -e 's/^#CCFLAG +=/CCFLAG +=/g' Makefile
sed -i.bak -e 's/^#CCFLAG =/CCFLAG =/g' Makefile

echo "----------------------------------"
cat Makefile
echo "----------------------------------"

make -j4

# at this point the make file leaves you in the python dir
cd ..
