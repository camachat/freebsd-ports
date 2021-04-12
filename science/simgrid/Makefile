<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	SimGrid
DISTVERSION=	3.26
=======
PORTNAME=	SimGrid
DISTVERSION=	3.27
>>>>>>> upstream/main
CATEGORIES=	science

MAINTAINER=	yuri@FreeBSD.org
COMMENT=	Simulator of the behavior of large-scale distributed systems

LICENSE=	LGPL21
LICENSE_FILE=	${WRKSRC}/LICENSE-LGPL-2.1

LIB_DEPENDS=	libboost_context.so:devel/boost-libs \
		libcgraph.so:graphics/graphviz \
		${LIB_DEPENDS_${ARCH}}
LIB_DEPENDS_aarch64=	libunwind.so:devel/libunwind
LIB_DEPENDS_amd64=	libunwind.so:devel/libunwind
LIB_DEPENDS_i386=	libunwind.so:devel/libunwind
LIB_DEPENDS_powerpc64=	libunwind.so:devel/libunwind
<<<<<<< HEAD

USES=		cmake:noninja compiler:c++11-lang perl5 python:3.6+ shebangfix
=======
LIB_DEPENDS_powerpc64le=	libunwind.so:devel/libunwind

USES=		cmake:noninja compiler:c++14-lang perl5 python:3.6+ shebangfix
>>>>>>> upstream/main
SHEBANG_FILES=	tools/simgrid_convert_TI_traces.py
USE_LDCONFIG=	yes

USE_GITLAB=	yes
GL_SITE=	https://framagit.org/
GL_ACCOUNT=	simgrid
GL_PROJECT=	simgrid
<<<<<<< HEAD
GL_COMMIT=	0e789e05d5e7bee542136978a4b4655c13240b56
=======
GL_COMMIT=	a585144d870def9053b47ed0f5cd92ac337038a5
>>>>>>> upstream/main

CMAKE_OFF=	enable_python

LDFLAGS+=	/usr/lib/liblzma.so # workaround for https://bugs.freebsd.org/bugzilla/show_bug.cgi?id=237919

<<<<<<< HEAD
INSTALL_TARGET=	install/strip

=======
>>>>>>> upstream/main
# it'll probably break on gcc-based systems, see https://framagit.org/simgrid/simgrid/issues/12, it'll need CMAKE_OFF=enable_lto and possibly more

.include <bsd.port.mk>
