extern crate bindgen;

use std::env;
use std::path::PathBuf;
use std::fs;

fn main() {

    let dirs = [
        "c/src/include",
        "c/src/vm",
        "c/src/optional"
    ];
    let src = {
        let mut src = Vec::with_capacity(32);
        for dir in dirs.iter().map(|d| fs::read_dir(d).unwrap()) {
            for entry in dir {
                let path = entry.unwrap().path();
                if path.extension().unwrap() == "c" {
                    src.push(path);
                }
            }
        }
        src
    };
    let mut builder = cc::Build::new();
    for d in &dirs {
        builder.include(d);
    }
    let build = builder
        .files(src.iter());
    build.compile("wren");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        .derive_debug(false)
        .whitelist_type("Wren.*")
        .whitelist_function("wren.*")
        .whitelist_var("WREN.*")

        .opaque_type("WrenVM")
        .opaque_type("WrenHandle")

        .prepend_enum_name(false)
        
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
