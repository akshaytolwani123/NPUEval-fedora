# Docker setup

The `install.sh` script in the parent directory will call these dockers in sequence.

We use the XRT docker as our base, which will install most Ubuntu dependencies, and build the debian packages for XRT and XDNA. These will be stored in the image at /XDNA/debs. To build this, from the parent directory, run

```
docker build -t xrt -f docker/Dockerfile.xrt .
```

To install the built packages on your host system you can copy them to your host workspace using

```
docker run --rm -v $(pwd)/docker:/host_dir xrt:latest bash -c "cp -v /XDNA/debs/*.deb /host_dir/"
```

then install, i.e.

```
sudo dpkg -i *.deb
```

The XDNA driver must be installed on your host which will insert the amdxdna kernel module - the same driver will be installed during npueval image build and without the kernel module present the build will fail.

Once the xrt docker is setup and the XDNA driver is installed on the host system you can build the npueval docker (run this from the same directory as install.sh):
```
docker build -t npueval -f docker/Dockerfile.npueval .
```
