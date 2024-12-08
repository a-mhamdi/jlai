DIRS=$(ls -la | grep '^d' | awk '{print $9}' | grep -v '^\.')

for dir in $DIRS; do
	cd ${WORKING_DIR}$dir
	rm Manifest.toml
	sed '/^ImageShow/d' Project.toml > tmp && cat tmp > Project.toml && rm tmp
	# sed -i '/^Image/d' Project.toml 
	# awk '!/^Image/' Project.toml > ${WORKING_DIR}/temp && mv ${WORKING_DIR}/temp ${WORKING_DIR}/Project.toml
	julia -e 'import Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.resolve(); Pkg.precompile()'
done
