# Make what is possible

COMMON     = -std=c++11 -O3 -g -Icifo4/include cifo4/src/main.cpp
NSIMD_URL  = "git@github.com:agenium-scale/nsimd.git"
NSIMD_URL2 = "https://github.com/agenium-scale/nsimd.git"
DATA_URL   = "ssh://git@phabricator2.numscale.com/diffusion/153/efispec-data.git"

all:
	echo "Nothing to do"

clean:
	rm -f bin/*
	rm -rf nsimd

get-nsimd:
	[ -e "nsimd/README.md" ] && \
            ( git -C nsimd pull ) || \
            ( git clone $(NSIMD_URL) || git clone $(NSIMD_URL2) )
	git -C nsimd checkout spmd
	( python3 --version 1>/dev/null 2>/dev/null && \
            python3 nsimd/egg/hatch.py -lf || \
            python nsimd/egg/hatch.py -lf )

get-data:
	[ -e "data/README.md" ] && ( git -C data pull ) \
	|| git clone $(DATA_URL) data

gcc-scalar:
	mkdir -p bin
	g++ $(COMMON) cifo4/src/cifo4-local.cpp -o bin/$@

gcc-neon128:
	mkdir -p bin
	arm-linux-gnueabihf-g++ -mfpu=neon -mfloat-abi=softp\
	$(COMMON) cifo4/src/cifo4-neon-local.cpp -o bin/$@

gcc-nsimd-neon128:
	mkdir -p bin
	arm-linux-gnueabihf-g++ -mfpu=neon -mfloat-abi=softp \
	$(COMMON) -DNEON128 cifo4/src/cifo4-neon-local.cpp -o bin/$@

gcc-aarch64:
	mkdir -p bin
	g++ $(COMMON) cifo4/src/cifo4-neon-local.cpp -o bin/$@

gcc-nsimd-aarch64:
	mkdir -p bin
	g++ -DAARCH64 -Insimd/include $(COMMON) \
	    cifo4/src/cifo4-nsimd-adv-local.cpp -o bin/$@

gcc-neon-nsimd2:
	mkdir -p bin
	g++ -DAARCH64 -Insimd/include $(COMMON) \
	    cifo4/src/cifo4-nsimd-adv-local-2.cpp -o bin/$@

gcc-sve512:
	mkdir -p bin
	g++ -march=armv8-a+sve -msve-vector-bits=512 $(COMMON) \
	    cifo4/src/cifo4-sve-local.cpp -o bin/$@

gcc-nsimd-sve512:
	mkdir -p bin
	g++ -DSVE512 -march=armv8-a+sve -msve-vector-bits=512 -Insimd/include \
	    $(COMMON) cifo4/src/cifo4-nsimd-adv-local.cpp -o bin/$@

clang-scalar:
	mkdir -p bin
	clang++ $(COMMON) cifo4/src/cifo4-local.cpp -o bin/$@

clang-aarch64:
	mkdir -p bin
	clang++ $(COMMON) cifo4/src/cifo4-neon-local.cpp -o bin/$@

clang-nsimd-aarch64:
	mkdir -p bin
	clang++ -DAARCH64 -Insimd/include $(COMMON) \
	    cifo4/src/cifo4-nsimd-adv-local.cpp -o bin/$@

clang-sve512:
	mkdir -p bin
	clang++ -march=armv8-a+sve -msve-vector-bits=512 $(COMMON) \
	    cifo4/src/cifo4-sve-local.cpp -o bin/$@

clang-nsimd-sve512:
	mkdir -p bin
	clang++ -DSVE512 -march=armv8-a+sve -msve-vector-bits=512 -Insimd/include \
	    $(COMMON) cifo4/src/cifo4-nsimd-adv-local.cpp -o bin/$@

armclang-scalar:
	mkdir -p bin
	armclang++ $(COMMON) cifo4/src/cifo4-local.cpp -o bin/$@

armclang-aarch64:
	mkdir -p bin
	armclang++ $(COMMON) cifo4/src/cifo4-neon-local.cpp -o bin/$@

armclang-nsimd-aarch64:
	mkdir -p bin
	armclang++ -DAARCH64 -Insimd/include $(COMMON) \
	           cifo4/src/cifo4-nsimd-adv-local.cpp -o bin/$@

armclang-sve512:
	mkdir -p bin
	armclang++ -march=armv8-a+sve $(COMMON) cifo4/src/cifo4-sve-local.cpp \
	           -o bin/$@

armclang-nsimd-sve512:
	mkdir -p bin
	armclang++ -DSVE -march=armv8-a+sve -Insimd/include \
	    $(COMMON) cifo4/src/cifo4-nsimd-adv-local.cpp -o bin/$@

