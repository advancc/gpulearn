#-- start of make_header -----------------

#====================================
#  Document SamplingSvc_python
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

cmt_SamplingSvc_python_has_no_target_tag = 1

#--------------------------------------------------------

ifdef cmt_SamplingSvc_python_has_target_tag

tags      = $(tag),$(CMTEXTRATAGS),target_SamplingSvc_python

SamplingSvc_tag = $(tag)

#cmt_local_tagfile_SamplingSvc_python = $(SamplingSvc_tag)_SamplingSvc_python.make
cmt_local_tagfile_SamplingSvc_python = $(bin)$(SamplingSvc_tag)_SamplingSvc_python.make

else

tags      = $(tag),$(CMTEXTRATAGS)

SamplingSvc_tag = $(tag)

#cmt_local_tagfile_SamplingSvc_python = $(SamplingSvc_tag).make
cmt_local_tagfile_SamplingSvc_python = $(bin)$(SamplingSvc_tag).make

endif

include $(cmt_local_tagfile_SamplingSvc_python)
#-include $(cmt_local_tagfile_SamplingSvc_python)

ifdef cmt_SamplingSvc_python_has_target_tag

cmt_final_setup_SamplingSvc_python = $(bin)setup_SamplingSvc_python.make
cmt_dependencies_in_SamplingSvc_python = $(bin)dependencies_SamplingSvc_python.in
#cmt_final_setup_SamplingSvc_python = $(bin)SamplingSvc_SamplingSvc_pythonsetup.make
cmt_local_SamplingSvc_python_makefile = $(bin)SamplingSvc_python.make

else

cmt_final_setup_SamplingSvc_python = $(bin)setup.make
cmt_dependencies_in_SamplingSvc_python = $(bin)dependencies.in
#cmt_final_setup_SamplingSvc_python = $(bin)SamplingSvcsetup.make
cmt_local_SamplingSvc_python_makefile = $(bin)SamplingSvc_python.make

endif

#cmt_final_setup = $(bin)setup.make
#cmt_final_setup = $(bin)SamplingSvcsetup.make

#SamplingSvc_python :: ;

dirs ::
	@if test ! -r requirements ; then echo "No requirements file" ; fi; \
	  if test ! -d $(bin) ; then $(mkdir) -p $(bin) ; fi

javadirs ::
	@if test ! -d $(javabin) ; then $(mkdir) -p $(javabin) ; fi

srcdirs ::
	@if test ! -d $(src) ; then $(mkdir) -p $(src) ; fi

help ::
	$(echo) 'SamplingSvc_python'

binobj = 
ifdef STRUCTURED_OUTPUT
binobj = SamplingSvc_python/
#SamplingSvc_python::
#	@if test ! -d $(bin)$(binobj) ; then $(mkdir) -p $(bin)$(binobj) ; fi
#	$(echo) "STRUCTURED_OUTPUT="$(bin)$(binobj)
endif

${CMTROOT}/src/Makefile.core : ;
ifdef use_requirements
$(use_requirements) : ;
endif

#-- end of make_header ------------------
#-- start of install_python_header ------


installarea = ${CMTINSTALLAREA}
install_python_dir = $(installarea)

ifneq ($(strip "$(source)"),"")
src = ../$(source)
dest = $(install_python_dir)/python
else
src = ../python
dest = $(install_python_dir)
endif

ifneq ($(strip "$(offset)"),"")
dest = $(install_python_dir)/python
endif

SamplingSvc_python :: SamplingSvc_pythoninstall

install :: SamplingSvc_pythoninstall

SamplingSvc_pythoninstall :: $(install_python_dir)
	@if [ ! "$(installarea)" = "" ] ; then\
	  echo "installation done"; \
	fi

$(install_python_dir) ::
	@if [ "$(installarea)" = "" ] ; then \
	  echo "Cannot install header files, no installation source specified"; \
	else \
	  if [ -d $(src) ] ; then \
	    echo "Installing files from $(src) to $(dest)" ; \
	    if [ "$(offset)" = "" ] ; then \
	      $(install_command) --exclude="*.py?" $(src) $(dest) ; \
	    else \
	      $(install_command) --exclude="*.py?" $(src) $(dest) --destname $(offset); \
	    fi ; \
	  else \
	    echo "no source  $(src)"; \
	  fi; \
	fi

SamplingSvc_pythonclean :: SamplingSvc_pythonuninstall

uninstall :: SamplingSvc_pythonuninstall

SamplingSvc_pythonuninstall ::
	@if test "$(installarea)" = ""; then \
	  echo "Cannot uninstall header files, no installation source specified"; \
	else \
	  echo "Uninstalling files from $(dest)"; \
	  $(uninstall_command) "$(dest)" ; \
	fi


#-- end of install_python_header ------
#-- start of cleanup_header --------------

clean :: SamplingSvc_pythonclean ;
#	@cd .

ifndef PEDANTIC
.DEFAULT::
	$(echo) "(SamplingSvc_python.make) $@: No rule for such target" >&2
else
.DEFAULT::
	$(error PEDANTIC: $@: No rule for such target)
endif

SamplingSvc_pythonclean ::
#-- end of cleanup_header ---------------
