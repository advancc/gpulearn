#-- start of make_header -----------------

#====================================
#  Library SamplingSvc
#
#   Generated Mon Nov 12 16:54:15 2018  by yiph
#
#====================================

include ${CMTROOT}/src/Makefile.core

ifdef tag
CMTEXTRATAGS = $(tag)
else
tag       = $(CMTCONFIG)
endif

cmt_SamplingSvc_has_no_target_tag = 1

#--------------------------------------------------------

ifdef cmt_SamplingSvc_has_target_tag

tags      = $(tag),$(CMTEXTRATAGS),target_SamplingSvc

SamplingSvc_tag = $(tag)

#cmt_local_tagfile_SamplingSvc = $(SamplingSvc_tag)_SamplingSvc.make
cmt_local_tagfile_SamplingSvc = $(bin)$(SamplingSvc_tag)_SamplingSvc.make

else

tags      = $(tag),$(CMTEXTRATAGS)

SamplingSvc_tag = $(tag)

#cmt_local_tagfile_SamplingSvc = $(SamplingSvc_tag).make
cmt_local_tagfile_SamplingSvc = $(bin)$(SamplingSvc_tag).make

endif

include $(cmt_local_tagfile_SamplingSvc)
#-include $(cmt_local_tagfile_SamplingSvc)

ifdef cmt_SamplingSvc_has_target_tag

cmt_final_setup_SamplingSvc = $(bin)setup_SamplingSvc.make
cmt_dependencies_in_SamplingSvc = $(bin)dependencies_SamplingSvc.in
#cmt_final_setup_SamplingSvc = $(bin)SamplingSvc_SamplingSvcsetup.make
cmt_local_SamplingSvc_makefile = $(bin)SamplingSvc.make

else

cmt_final_setup_SamplingSvc = $(bin)setup.make
cmt_dependencies_in_SamplingSvc = $(bin)dependencies.in
#cmt_final_setup_SamplingSvc = $(bin)SamplingSvcsetup.make
cmt_local_SamplingSvc_makefile = $(bin)SamplingSvc.make

endif

#cmt_final_setup = $(bin)setup.make
#cmt_final_setup = $(bin)SamplingSvcsetup.make

#SamplingSvc :: ;

dirs ::
	@if test ! -r requirements ; then echo "No requirements file" ; fi; \
	  if test ! -d $(bin) ; then $(mkdir) -p $(bin) ; fi

javadirs ::
	@if test ! -d $(javabin) ; then $(mkdir) -p $(javabin) ; fi

srcdirs ::
	@if test ! -d $(src) ; then $(mkdir) -p $(src) ; fi

help ::
	$(echo) 'SamplingSvc'

binobj = 
ifdef STRUCTURED_OUTPUT
binobj = SamplingSvc/
#SamplingSvc::
#	@if test ! -d $(bin)$(binobj) ; then $(mkdir) -p $(bin)$(binobj) ; fi
#	$(echo) "STRUCTURED_OUTPUT="$(bin)$(binobj)
endif

${CMTROOT}/src/Makefile.core : ;
ifdef use_requirements
$(use_requirements) : ;
endif

#-- end of make_header ------------------
#-- start of libary_header ---------------

SamplingSvclibname   = $(bin)$(library_prefix)SamplingSvc$(library_suffix)
SamplingSvclib       = $(SamplingSvclibname).a
SamplingSvcstamp     = $(bin)SamplingSvc.stamp
SamplingSvcshstamp   = $(bin)SamplingSvc.shstamp

SamplingSvc :: dirs  SamplingSvcLIB
	$(echo) "SamplingSvc ok"

cmt_SamplingSvc_has_prototypes = 1

#--------------------------------------

ifdef cmt_SamplingSvc_has_prototypes

SamplingSvcprototype :  ;

endif

SamplingSvccompile : $(bin)SamplingSvc.o ;

#-- end of libary_header ----------------
#-- start of libary ----------------------

SamplingSvcLIB :: $(SamplingSvclib) $(SamplingSvcshstamp)
	$(echo) "SamplingSvc : library ok"

$(SamplingSvclib) :: $(bin)SamplingSvc.o
	$(lib_echo) "static library $@"
	$(lib_silent) [ ! -f $@ ] || \rm -f $@
	$(lib_silent) $(ar) $(SamplingSvclib) $(bin)SamplingSvc.o
	$(lib_silent) $(ranlib) $(SamplingSvclib)
	$(lib_silent) cat /dev/null >$(SamplingSvcstamp)

#------------------------------------------------------------------
#  Future improvement? to empty the object files after
#  storing in the library
#
##	  for f in $?; do \
##	    rm $${f}; touch $${f}; \
##	  done
#------------------------------------------------------------------

#
# We add one level of dependency upon the true shared library 
# (rather than simply upon the stamp file)
# this is for cases where the shared library has not been built
# while the stamp was created (error??) 
#

$(SamplingSvclibname).$(shlibsuffix) :: $(SamplingSvclib) requirements $(use_requirements) $(SamplingSvcstamps)
	$(lib_echo) "shared library $@"
	$(lib_silent) if test "$(makecmd)"; then QUIET=; else QUIET=1; fi; QUIET=$${QUIET} bin="$(bin)" ld="$(shlibbuilder)" ldflags="$(shlibflags)" suffix=$(shlibsuffix) libprefix=$(library_prefix) libsuffix=$(library_suffix) $(make_shlib) "$(tags)" SamplingSvc $(SamplingSvc_shlibflags)
	$(lib_silent) cat /dev/null >$(SamplingSvcshstamp)

$(SamplingSvcshstamp) :: $(SamplingSvclibname).$(shlibsuffix)
	$(lib_silent) if test -f $(SamplingSvclibname).$(shlibsuffix) ; then cat /dev/null >$(SamplingSvcshstamp) ; fi

SamplingSvcclean ::
	$(cleanup_echo) objects SamplingSvc
	$(cleanup_silent) /bin/rm -f $(bin)SamplingSvc.o
	$(cleanup_silent) /bin/rm -f $(patsubst %.o,%.d,$(bin)SamplingSvc.o) $(patsubst %.o,%.dep,$(bin)SamplingSvc.o) $(patsubst %.o,%.d.stamp,$(bin)SamplingSvc.o)
	$(cleanup_silent) cd $(bin); /bin/rm -rf SamplingSvc_deps SamplingSvc_dependencies.make

#-----------------------------------------------------------------
#
#  New section for automatic installation
#
#-----------------------------------------------------------------

install_dir = ${CMTINSTALLAREA}/$(tag)/lib
SamplingSvcinstallname = $(library_prefix)SamplingSvc$(library_suffix).$(shlibsuffix)

SamplingSvc :: SamplingSvcinstall ;

install :: SamplingSvcinstall ;

SamplingSvcinstall :: $(install_dir)/$(SamplingSvcinstallname)
ifdef CMTINSTALLAREA
	$(echo) "installation done"
endif

$(install_dir)/$(SamplingSvcinstallname) :: $(bin)$(SamplingSvcinstallname)
ifdef CMTINSTALLAREA
	$(install_silent) $(cmt_install_action) \
	    -source "`(cd $(bin); pwd)`" \
	    -name "$(SamplingSvcinstallname)" \
	    -out "$(install_dir)" \
	    -cmd "$(cmt_installarea_command)" \
	    -cmtpath "$($(package)_cmtpath)"
endif

##SamplingSvcclean :: SamplingSvcuninstall

uninstall :: SamplingSvcuninstall ;

SamplingSvcuninstall ::
ifdef CMTINSTALLAREA
	$(cleanup_silent) $(cmt_uninstall_action) \
	    -source "`(cd $(bin); pwd)`" \
	    -name "$(SamplingSvcinstallname)" \
	    -out "$(install_dir)" \
	    -cmtpath "$($(package)_cmtpath)"
endif

#-- end of libary -----------------------
#-- start of dependencies ------------------
ifneq ($(MAKECMDGOALS),SamplingSvcclean)
ifneq ($(MAKECMDGOALS),uninstall)
ifneq ($(MAKECMDGOALS),SamplingSvcprototype)

$(bin)SamplingSvc_dependencies.make : $(use_requirements) $(cmt_final_setup_SamplingSvc)
	$(echo) "(SamplingSvc.make) Rebuilding $@"; \
	  $(build_dependencies) -out=$@ -start_all $(src)SamplingSvc.cc -end_all $(includes) $(app_SamplingSvc_cppflags) $(lib_SamplingSvc_cppflags) -name=SamplingSvc $? -f=$(cmt_dependencies_in_SamplingSvc) -without_cmt

-include $(bin)SamplingSvc_dependencies.make

endif
endif
endif

SamplingSvcclean ::
	$(cleanup_silent) \rm -rf $(bin)SamplingSvc_deps $(bin)SamplingSvc_dependencies.make
#-- end of dependencies -------------------
#-- start of cpp_library -----------------

ifneq (,)

ifneq ($(MAKECMDGOALS),SamplingSvcclean)
ifneq ($(MAKECMDGOALS),uninstall)
-include $(bin)$(binobj)SamplingSvc.d

$(bin)$(binobj)SamplingSvc.d :

$(bin)$(binobj)SamplingSvc.o : $(cmt_final_setup_SamplingSvc)

$(bin)$(binobj)SamplingSvc.o : $(src)SamplingSvc.cc
	$(cpp_echo) $(src)SamplingSvc.cc
	$(cpp_silent) $(cppcomp)  -o $@ $(use_pp_cppflags) $(SamplingSvc_pp_cppflags) $(lib_SamplingSvc_pp_cppflags) $(SamplingSvc_pp_cppflags) $(use_cppflags) $(SamplingSvc_cppflags) $(lib_SamplingSvc_cppflags) $(SamplingSvc_cppflags) $(SamplingSvc_cc_cppflags)  $(src)SamplingSvc.cc
endif
endif

else
$(bin)SamplingSvc_dependencies.make : $(SamplingSvc_cc_dependencies)

$(bin)SamplingSvc_dependencies.make : $(src)SamplingSvc.cc

$(bin)$(binobj)SamplingSvc.o : $(SamplingSvc_cc_dependencies)
	$(cpp_echo) $(src)SamplingSvc.cc
	$(cpp_silent) $(cppcomp) -o $@ $(use_pp_cppflags) $(SamplingSvc_pp_cppflags) $(lib_SamplingSvc_pp_cppflags) $(SamplingSvc_pp_cppflags) $(use_cppflags) $(SamplingSvc_cppflags) $(lib_SamplingSvc_cppflags) $(SamplingSvc_cppflags) $(SamplingSvc_cc_cppflags)  $(src)SamplingSvc.cc

endif

#-- end of cpp_library ------------------
#-- start of cleanup_header --------------

clean :: SamplingSvcclean ;
#	@cd .

ifndef PEDANTIC
.DEFAULT::
	$(echo) "(SamplingSvc.make) $@: No rule for such target" >&2
else
.DEFAULT::
	$(error PEDANTIC: $@: No rule for such target)
endif

SamplingSvcclean ::
#-- end of cleanup_header ---------------
#-- start of cleanup_library -------------
	$(cleanup_echo) library SamplingSvc
	-$(cleanup_silent) cd $(bin) && \rm -f $(library_prefix)SamplingSvc$(library_suffix).a $(library_prefix)SamplingSvc$(library_suffix).$(shlibsuffix) SamplingSvc.stamp SamplingSvc.shstamp
#-- end of cleanup_library ---------------
