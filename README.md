## The Brightness Clustering Transform and Locally Contrasting Keypoints
___

This code uses opencv 3 libraries


### Compilation

CMake and basic compilation tools are needed for this to work.

```
git clone https://github.com/jimmylomro/locky.git
cd locky
mkdir build
cd build
cmake ..
make
```

If the compilation goes well, directories with binaries and static link libraries should appear.


### Usage

- This code generates a static link library with the code for obtaining the LOCKY or the LOCKYS features. The parameters are straight forward according to the paper.
- The last parameter 'bool bcts = true' sets the type of BCT to use. Set to 'true' for the BCTS as shown in [this paper][2], or to 'false' to use the regular BCT as shown in [this paper][1]


This code has only been tested in OSX 10.11

Credits for the image im.jpg to Juan Romano

[1]: http://link.springer.com/chapter/10.1007/978-3-319-23192-1_30
[2]: http://link.springer.com/article/10.1007/s00138-016-0785-3

___

#### Author:
Jaime Lomeli-R. (2016)
University of Southampton
jaime.lomeli.rodriguez@gmail.com
