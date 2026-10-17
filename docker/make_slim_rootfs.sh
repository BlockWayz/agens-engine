#!/bin/bash
# Slim runtime image root filesystem, assembled inside the build image (which has the CUDA 12.9 + libtorch runtime):
# the engine (dist: bin/agens, bin/agens-entry.sh, lib/*.so, license files) + the shared-library closure of the engine
# (libtorch, cuBLAS, cudart, ... resolved with ldd) + bash and glibc. The NVIDIA container toolkit provides the
# driver libraries (libcuda, the PTX JIT) at run time.
#   make_slim_rootfs.sh <dist dir> <out rootfs.tar>       (run in the build image; then `docker import`)
set -euo pipefail
DIST=$1; OUT=$2; R=$(mktemp -d)
mkdir -p $R/opt/agens/lib $R/opt/agens/bin $R/usr/lib/x86_64-linux-gnu $R/usr/bin $R/usr/sbin $R/tmp $R/etc $R/lib64
chmod 1777 $R/tmp
cp -a $DIST/. $R/opt/agens/
closure() { LD_LIBRARY_PATH=$R/opt/agens/lib ldd "$@" 2>/dev/null | awk '/=> \//{print $3}' | sort -u; }
for lib in $(closure $R/opt/agens/bin/agens $R/opt/agens/lib/*.so); do
  case $lib in
    $R/*) ;;
    /lib/x86_64-linux-gnu/*|/usr/lib/x86_64-linux-gnu/*) cp -L $lib $R/usr/lib/x86_64-linux-gnu/;;
    *) cp -L $lib $R/opt/agens/lib/;;
  esac
done
# second pass: dependencies of the copied CUDA / torch libraries themselves
for lib in $(closure $R/opt/agens/lib/*.so*); do
  [ -e $R/opt/agens/lib/$(basename $lib) ] || [ -e $R/usr/lib/x86_64-linux-gnu/$(basename $lib) ] || {
    case $lib in /lib/x86_64-linux-gnu/*|/usr/lib/x86_64-linux-gnu/*) cp -L $lib $R/usr/lib/x86_64-linux-gnu/;; *) cp -L $lib $R/opt/agens/lib/;; esac; }
done
cp -L /bin/bash $R/usr/bin/bash
for lib in $(closure /bin/bash); do cp -L $lib $R/usr/lib/x86_64-linux-gnu/; done
cp -L /lib64/ld-linux-x86-64.so.2 $R/lib64/
cp -a /sbin/ldconfig* $R/usr/sbin/ 2>/dev/null || true
ln -s usr/bin $R/bin; ln -s usr/lib $R/lib; ln -s usr/sbin $R/sbin
printf 'root:x:0:0:root:/root:/bin/bash\n' > $R/etc/passwd; printf 'root:x:0:\n' > $R/etc/group
echo /opt/agens/lib > $R/etc/ld.so.conf
strip --strip-unneeded $R/opt/agens/lib/libagens_*.so 2>/dev/null || true
tar -C $R -cf $OUT . && rm -rf $R
du -h $OUT
