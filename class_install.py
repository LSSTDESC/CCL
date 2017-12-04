import os

print "Downloading class..."
os.system('wget https://github.com/lesgourg/class_public/archive/v2.6.3.tar.gz')
print "Unpacking..."
os.system('tar -xvf v2.6.3.tar.gz >> log_class_install ')
os.system('mv class_public-2.6.3 class')
print "Compiling..."
comp_string='cd class; '
comp_string+='make >> ../log_class_install ; '
comp_string+='gcc -shared -o libclass.so -Wl,--whole-archive libclass.a -Wl,--no-whole-archive -lgomp >> ./log_class_install ; '
comp_string+='ls; '
os.system(comp_string)
print "Cleanup"
os.system('rm v2.6.3.tar.gz')
