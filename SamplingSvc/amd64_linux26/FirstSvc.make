#-- start of make_header -----------------

#====================================
#  Library FirstSvc
#
#   Generated Tue Oct 30 17:37:36 2018  by zoujh
#
#====================================

include ${CMTROOT}/src/Makefile.core

ifdef tag
CMTEXTRATAGS = $(tag)
else
tag       = $(CMTCONFIG)
endif

cmt_FirstSvc_has_no_target_tag = 1

#--------------------------------------------------------

ifdef cmt_FirstSvc_has_target_tag

tags      = $(tag),$(CMTEXTRATAGS),target_FirstSvc

FirstSvc_tag = $(tag)

#cmt_local_tagfile_FirstSvc = $(FirstSvc_tag)_FirstSvc.make
cmt_local_tagfile_FirstSvc = $(bin)$(FirstSvc_tag)_FirstSvc.make

else

tags      = $(tag),$(CMTEXTRATAGS)

FirstSvc_tag = $(tag)

#cmt_local_tagfile_FirstSvc = $(FirstSvc_tag).make
cmt_local_tagfile_FirstSvc = $(bin)$(FirstSvc_tag).make

endif

include $(cmt_local_tagfile_FirstSvc)
#-include $(cmt_local_tagfile_FirstSvc)

ifdef cmt_FirstSvc_has_target_tag

cmt_final_setup_FirstSvc = $(bin)setup_FirstSvc.make
cmt_dependencies_in_FirstSvc = $(bin)dependencies_FirstSvc.in
#cmt_final_setup_FirstSvc = $(bin)FirstSvc_FirstSvcsetup.make
cmt_local_FirstSvc_makefile = $(bin)FirstSvc.make

else

cmt_final_setup_FirstSvc = $(bin)setup.make
cmt_dependencies_in_FirstSvc = $(bin)dependencies.in
#cmt_final_setup_FirstSvc = $(bin)FirstSvcsetup.make
cmt_local_FirstSvc_makefile = $(bin)FirstSvc.make

endif

#cmt_final_setup = $(bin)setup.make
#cmt_final_setup = $(bin)FirstSvcsetup.make

#FirstSvc :: ;

dirs ::
	@if test ! -r requirements ; then echo "No requirements file" ; fi; \
	  if test ! -d $(bin) ; then $(mkdir) -p $(bin) ; fi

javadirs ::
	@if test ! -d $(javabin) ; then $(mkdir) -p $(javabin) ; fi

srcdirs ::
	@if test ! -d $(src) ; then $(mkdir) -p $(src) ; fi

help ::
	$(echo) 'FirstSvc'

binobj = 
ifdef STRUCTURED_OUTPUT
binobj = FirstSvc/
#FirstSvc::
#	@if test ! -d $(bin)$(binobj) ; then $(mkdir) -p $(bin)$(binobj) ; fi
#	$(echo) "STRUCTURED_OUTPUT="$(bin)$(binobj)
endif

${CMTROOT}/src/Makefile.core : ;
ifdef use_requirements
$(use_requirements) : ;
endif

#-- end of make_header ------------------
#-- start of libary_header ---------------

FirstSvclibname   = $(bin)$(library_prefix)FirstSvc$(library_suffix)
FirstSvclib       = $(FirstSvclibname).a
FirstSvcstamp     = $(bin)FirstSvc.stamp
FirstSvcshstamp   = $(bin)FirstSvc.shstamp

FirstSvc :: dirs  FirstSvcLIB
	$(echo) "FirstSvc ok"

cmt_FirstSvc_has_prototypes = 1

#--------------------------------------

ifdef cmt_FirstSvc_has_prototypes

FirstSvcprototype :  ;

endif

FirstSvccompile : $(bin)FirstSvc.o ;

#-- end of libary_header ----------------
#-- start of libary ----------------------

FirstSvcLIB :: $(FirstSvclib) $(FirstSvcshstamp)
	$(echo) "FirstSvc : library ok"

$(FirstSvclib) :: $(bin)FirstSvc.o
	$(lib_echo) "static library $@"
	$(lib_silent) [ ! -f $@ ] || \rm -f $@
	$(lib_silent) $(ar) $(FirstSvclib) $(bin)FirstSvc.o
	$(lib_silent) $(ranlib) $(FirstSvclib)
	$(lib_silent) cat /dev/null >$(FirstSvcstamp)

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

$(FirstSvclibname).$(shlibsuffix) :: $(FirstSvclib) requirements $(use_requirements) $(FirstSvcstamps)
	$(lib_echo) "shared library $@"
	$(lib_silent) if test "$(makecmd)"; then QUIET=; else QUIET=1; fi; QUIET=$${QUIET} bin="$(bin)" ld="$(shlibbuilder)" ldflags="$(shlibflags)" suffix=$(shlibsuffix) libprefix=$(library_prefix) libsuffix=$(library_suffix) $(make_shlib) "$(tags)" FirstSvc $(FirstSvc_shlibflags)
	$(lib_silent) cat /dev/null >$(FirstSvcshstamp)

$(FirstSvcshstamp) :: $(FirstSvclibname).$(shlibsuffix)
	$(lib_silent) if test -f $(FirstSvclibname).$(shlibsuffix) ; then cat /dev/null >$(FirstSvcshstamp) ; fi

FirstSvcclean ::
	$(cleanup_echo) objects FirstSvc
	$(cleanup_silent) /bin/rm -f $(bin)FirstSvc.o
	$(cleanup_silent) /bin/rm -f $(patsubst %.o,%.d,$(bin)FirstSvc.o) $(patsubst %.o,%.dep,$(bin)FirstSvc.o) $(patsubst %.o,%.d.stamp,$(bin)FirstSvc.o)
	$(cleanup_silent) cd $(bin); /bin/rm -rf FirstSvc_deps FirstSvc_dependencies.make

#-----------------------------------------------------------------
#
#  New section for automatic installation
#
#-----------------------------------------------------------------

install_dir = ${CMTINSTALLAREA}/$(tag)/lib
FirstSvcinstallname = $(library_prefix)FirstSvc$(library_suffix).$(shlibsuffix)

FirstSvc :: FirstSvcinstall ;

install :: FirstSvcinstall ;

FirstSvcinstall :: $(install_dir)/$(FirstSvcinstallname)
ifdef CMTINSTALLAREA
	$(echo) "installation done"
endif

$(install_dir)/$(FirstSvcinstallname) :: $(bin)$(FirstSvcinstallname)
ifdef CMTINSTALLAREA
	$(install_silent) $(cmt_install_action) \
	    -source "`(cd $(bin); pwd)`" \
	    -name "$(FirstSvcinstallname)" \
	    -out "$(install_dir)" \
	    -cmd "$(cmt_installarea_command)" \
	    -cmtpath "$($(package)_cmtpath)"
endif

##FirstSvcclean :: FirstSvcuninstall

uninstall :: FirstSvcuninstall ;

FirstSvcuninstall ::
ifdef CMTINSTALLAREA
	$(cleanup_silent) $(cmt_uninstall_action) \
	    -source "`(cd $(bin); pwd)`" \
	    -name "$(FirstSvcinstallname)" \
	    -out "$(install_dir)" \
	    -cmtpath "$($(package)_cmtpath)"
endif

#-- end of libary -----------------------
#-- start of dependencies ------------------
ifneq ($(MAKECMDGOALS),FirstSvcclean)
ifneq ($(MAKECMDGOALS),uninstall)
ifneq ($(MAKECMDGOALS),FirstSvcprototype)

$(bin)FirstSvc_dependencies.make : $(use_requirements) $(cmt_final_setup_FirstSvc)
	$(echo) "(FirstSvc.make) Rebuilding $@"; \
	  $(build_dependencies) -out=$@ -start_all $(src)FirstSvc.cc -end_all $(includes) $(app_FirstSvc_cppflags) $(lib_FirstSvc_cppflags) -name=FirstSvc $? -f=$(cmt_dependencies_in_FirstSvc) -without_cmt

-include $(bin)FirstSvc_dependencies.make

endif
endif
endif

FirstSvcclean ::
	$(cleanup_silent) \rm -rf $(bin)FirstSvc_deps $(bin)FirstSvc_dependencies.make
#-- end of dependencies -------------------
#-- start of cpp_library -----------------

ifneq (,)

ifneq ($(MAKECMDGOALS),FirstSvcclean)
ifneq ($(MAKECMDGOALS),uninstall)
-include $(bin)$(binobj)FirstSvc.d

$(bin)$(binobj)FirstSvc.d :

$(bin)$(binobj)FirstSvc.o : $(cmt_final_setup_FirstSvc)

$(bin)$(binobj)FirstSvc.o : $(src)FirstSvc.cc
	$(cpp_echo) $(src)FirstSvc.cc
	$(cpp_silent) $(cppcomp)  -o $@ $(use_pp_cppflags) $(FirstSvc_pp_cppflags) $(lib_FirstSvc_pp_cppflags) $(FirstSvc_pp_cppflags) $(use_cppflags) $(FirstSvc_cppflags) $(lib_FirstSvc_cppflags) $(FirstSvc_cppflags) $(FirstSvc_cc_cppflags)  $(src)FirstSvc.cc
endif
endif

else
$(bin)FirstSvc_dependencies.make : $(FirstSvc_cc_dependencies)

$(bin)FirstSvc_dependencies.make : $(src)FirstSvc.cc

$(bin)$(binobj)FirstSvc.o : $(FirstSvc_cc_dependencies)
	$(cpp_echo) $(src)FirstSvc.cc
	$(cpp_silent) $(cppcomp) -o $@ $(use_pp_cppflags) $(FirstSvc_pp_cppflags) $(lib_FirstSvc_pp_cppflags) $(FirstSvc_pp_cppflags) $(use_cppflags) $(FirstSvc_cppflags) $(lib_FirstSvc_cppflags) $(FirstSvc_cppflags) $(FirstSvc_cc_cppflags)  $(src)FirstSvc.cc

endif

#-- end of cpp_library ------------------
#-- start of cleanup_header --------------

clean :: FirstSvcclean ;
#	@cd .

ifndef PEDANTIC
.DEFAULT::
	$(echo) "(FirstSvc.make) $@: No rule for such target" >&2
else
.DEFAULT::
	$(error PEDANTIC: $@: No rule for such target)
endif

FirstSvcclean ::
#-- end of cleanup_header ---------------
#-- start of cleanup_library -------------
	$(cleanup_echo) library FirstSvc
	-$(cleanup_silent) cd $(bin) && \rm -f $(library_prefix)FirstSvc$(library_suffix).a $(library_prefix)FirstSvc$(library_suffix).$(shlibsuffix) FirstSvc.stamp FirstSvc.shstamp
#-- end of cleanup_library ---------------
