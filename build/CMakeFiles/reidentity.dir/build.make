# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk/build

# Include any dependencies generated for this target.
include CMakeFiles/reidentity.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/reidentity.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reidentity.dir/flags.make

CMakeFiles/reidentity.dir/src/main.cc.o: CMakeFiles/reidentity.dir/flags.make
CMakeFiles/reidentity.dir/src/main.cc.o: ../src/main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/reidentity.dir/src/main.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/reidentity.dir/src/main.cc.o -c /home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk/src/main.cc

CMakeFiles/reidentity.dir/src/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reidentity.dir/src/main.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk/src/main.cc > CMakeFiles/reidentity.dir/src/main.cc.i

CMakeFiles/reidentity.dir/src/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reidentity.dir/src/main.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk/src/main.cc -o CMakeFiles/reidentity.dir/src/main.cc.s

CMakeFiles/reidentity.dir/src/main.cc.o.requires:

.PHONY : CMakeFiles/reidentity.dir/src/main.cc.o.requires

CMakeFiles/reidentity.dir/src/main.cc.o.provides: CMakeFiles/reidentity.dir/src/main.cc.o.requires
	$(MAKE) -f CMakeFiles/reidentity.dir/build.make CMakeFiles/reidentity.dir/src/main.cc.o.provides.build
.PHONY : CMakeFiles/reidentity.dir/src/main.cc.o.provides

CMakeFiles/reidentity.dir/src/main.cc.o.provides.build: CMakeFiles/reidentity.dir/src/main.cc.o


# Object files for target reidentity
reidentity_OBJECTS = \
"CMakeFiles/reidentity.dir/src/main.cc.o"

# External object files for target reidentity
reidentity_EXTERNAL_OBJECTS =

reidentity: CMakeFiles/reidentity.dir/src/main.cc.o
reidentity: CMakeFiles/reidentity.dir/build.make
reidentity: ../lib/libdragon_cc.so
reidentity: ../lib/libdragon_reid.so
reidentity: CMakeFiles/reidentity.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable reidentity"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reidentity.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reidentity.dir/build: reidentity

.PHONY : CMakeFiles/reidentity.dir/build

CMakeFiles/reidentity.dir/requires: CMakeFiles/reidentity.dir/src/main.cc.o.requires

.PHONY : CMakeFiles/reidentity.dir/requires

CMakeFiles/reidentity.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reidentity.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reidentity.dir/clean

CMakeFiles/reidentity.dir/depend:
	cd /home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk /home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk /home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk/build /home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk/build /home/disk1/xupeichao/REID_SDK_DEVELOP/user_sdk/build/CMakeFiles/reidentity.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/reidentity.dir/depend

