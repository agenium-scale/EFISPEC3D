# Make what is possible

COMMON     = -std=c++11 -O3 -g -Icifo4/include cifo4/src/main.cpp
NSIMD_URL  = "git@github.com:agenium-scale/nsimd.git"
NSIMD_URL2 = "https://github.com/agenium-scale/nsimd.git"

all: get-nsimd
	-make -s gcc-scalar
	-make -s gcc-neon-intrinsics
	-make -s gcc-neon-nsimd
	-make -s gcc-sve-intrinsics
	-make -s gcc-sve-nsimd
	-make -s armclang-scalar
	-make -s armclang-neon-intrinsics
	-make -s armclang-neon-nsimd
	-make -s armclang-sve-intrinsics
	-make -s armclang-sve-nsimd
	@for i in `ls bin`; do echo $${i}; ./bin/$${i}; done

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

gcc-scalar:
	mkdir -p bin
	g++ $(COMMON) cifo4/src/cifo4-local.cpp -o bin/$@

gcc-neon-intrinsics:
	mkdir -p bin
	g++ $(COMMON) cifo4/src/cifo4-neon-local.cpp -o bin/$@

gcc-neon-nsimd:
	mkdir -p bin
	g++ -DAARCH64 -Insimd/include $(COMMON) \
	    cifo4/src/cifo4-nsimd-adv-local.cpp -o bin/$@

gcc-neon-nsimd2:
	mkdir -p bin
	g++ -DAARCH64 -Insimd/include $(COMMON) \
	    cifo4/src/cifo4-nsimd-adv-local-2.cpp -o bin/$@

gcc-sve-intrinsics:
	mkdir -p bin
	g++ -march=armv8-a+sve -msve-vector-bits=512 $(COMMON) \
	    cifo4/src/cifo4-sve-local.cpp -o bin/$@

gcc-sve-nsimd:
	mkdir -p bin
	g++ -DSVE512 -march=armv8-a+sve -msve-vector-bits=512 -Insimd/include \
	    $(COMMON) cifo4/src/cifo4-nsimd-adv-local.cpp -o bin/$@

clang-scalar:
	mkdir -p bin
	clang++ $(COMMON) cifo4/src/cifo4-local.cpp -o bin/$@

clang-neon-intrinsics:
	mkdir -p bin
	clang++ $(COMMON) cifo4/src/cifo4-neon-local.cpp -o bin/$@

clang-neon-nsimd:
	mkdir -p bin
	clang++ -DAARCH64 -Insimd/include $(COMMON) \
	    cifo4/src/cifo4-nsimd-adv-local.cpp -o bin/$@

clang-sve-intrinsics:
	mkdir -p bin
	clang++ -march=armv8-a+sve -msve-vector-bits=512 $(COMMON) \
	    cifo4/src/cifo4-sve-local.cpp -o bin/$@

clang-sve-nsimd:
	mkdir -p bin
	clang++ -DSVE512 -march=armv8-a+sve -msve-vector-bits=512 -Insimd/include \
	    $(COMMON) cifo4/src/cifo4-nsimd-adv-local.cpp -o bin/$@

armclang-scalar:
	mkdir -p bin
	armclang++ $(COMMON) cifo4/src/cifo4-local.cpp -o bin/$@

armclang-neon-intrinsics:
	mkdir -p bin
	armclang++ $(COMMON) cifo4/src/cifo4-neon-local.cpp -o bin/$@

armclang-neon-nsimd:
	mkdir -p bin
	armclang++ -DAARCH64 -Insimd/include $(COMMON) \
	           cifo4/src/cifo4-nsimd-adv-local.cpp -o bin/$@

armclang-sve-intrinsics:
	mkdir -p bin
	armclang++ -march=armv8-a+sve $(COMMON) cifo4/src/cifo4-sve-local.cpp \
	           -o bin/$@

armclang-sve-nsimd:
	mkdir -p bin
	armclang++ -DSVE -march=armv8-a+sve -Insimd/include \
	    $(COMMON) cifo4/src/cifo4-nsimd-adv-local.cpp -o bin/$@

