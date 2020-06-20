# Make what is possible

COMMON     = -std=c++11 -O3 -Icifo4/include cifo4/src/main.cpp
NSIMD_URL  = "git@github.com:agenium-scale/nsimd.git"
NSIMD_URL2 = "https://github.com/agenium-scale/nsimd.git"

all:
	-make -s bin/gcc-scalar
	-make -s bin/gcc-neon-intrinsics
	-make -s bin/gcc-neon-nsimd
	-make -s bin/gcc-sve-intrinsics
	-make -s bin/gcc-sve-nsimd
	-make -s bin/armclang-scalar
	-make -s bin/armclang-neon-intrinsics
	-make -s bin/armclang-neon-nsimd
	-make -s bin/armclang-sve-intrinsics
	-make -s bin/armclang-sve-nsimd
	@for i in bin; do echo $${i}; ./$${i}; done

clean:
	rm -f bin/*
	
get-nsimd:
	[ -e "nsimd/README.md" ] && \
            ( git -C nsimd pull ) || \
            ( git clone $(NSIMD_URL) || git clone $(NSIMD_URL2) )
	(cd nsimd && ( python3 --version 1>/dev/null 2>/dev/null && \
            python3 nsimd/egg/hatch.py -lf || \
            python nsimd/egg/hatch.py -lf ) )

bin/gcc-scalar:
	mkdir -p bin
	g++ $(COMMON) cifo4/src/cifo4-local.cpp -o $@

bin/gcc-neon-intrinsics:
	mkdir -p bin
	g++ $(COMMON) cifo4/src/cifo4-neon-local.cpp -o $@

bin/gcc-neon-nsimd: get-nsimd
	mkdir -p bin
	g++ -DAARCH64 -Insimd/include $(COMMON) \
	    cifo4/src/cifo4-nsimd-adv-local.cpp -o $@

bin/gcc-sve-intrinsics:
	mkdir -p bin
	g++ -march=armv8-a+sve -msve-vector-bits=512 $(COMMON) \
	    cifo4/src/cifo4-sve-local.cpp -o $@

bin/gcc-sve-nsimd: get-nsimd
	mkdir -p bin
	g++ -DSVE512 -march=armv8-a+sve -msve-vector-bits=512 -Insimd/include \
	    $(COMMON) cifo4/src/cifo4-nsimd-adv-local.cpp -o $@

bin/armclang-scalar:
	mkdir -p bin
	armclang++ $(COMMON) cifo4/src/cifo4-local.cpp -o $@

bin/armclang-neon-intrinsics:
	mkdir -p bin
	armclang++ $(COMMON) cifo4/src/cifo4-neon-local.cpp -o $@

bin/armclang-neon-nsimd: get-nsimd
	mkdir -p bin
	armclang++ -DAARCH64 -Insimd/include $(COMMON) \
	           cifo4/src/cifo4-nsimd-adv-local.cpp -o $@

bin/armclang-sve-intrinsics:
	mkdir -p bin
	armclang++ -march=armv8-a+sve $(COMMON) cifo4/src/cifo4-sve-local.cpp \
	           -o $@

bin/armclang-sve-nsimd: get-nsimd
	mkdir -p bin
	armclang++ -DSVE -march=armv8-a+sve -Insimd/include \
	    $(COMMON) cifo4/src/cifo4-nsimd-adv-local.cpp -o $@

