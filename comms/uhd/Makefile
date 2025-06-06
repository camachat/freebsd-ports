PORTNAME=	uhd
DISTVERSIONPREFIX=	v
DISTVERSION=	4.8.0.0
PORTREVISION=	1
CATEGORIES=	comms hamradio
DIST_SUBDIR=	${PORTNAME}

MAINTAINER=	mr@FreeBSD.org
COMMENT=	Ettus Research UHD driver framework
WWW=		https://www.ettus.com/sdr-software/uhd-usrp-hardware-driver/

# See https://www.ettus.com/sdr-software/licenses/.  Dual license
# under GPLv3 and and "Alternate License" whose terms are not documented
# on the web site (contact info@ettus.com).  Mark as GPLv3 for now
# until terms of the "Alternate License" are made clear.  When that
# is understood and if appropriate, we can add that license here.
LICENSE=	GPLv3

LIB_DEPENDS=	${PY_BOOST} \
		libboost_system.so:devel/boost-libs
BUILD_DEPENDS=	flatc:devel/flatbuffers \
		orcc:devel/orc \
		cheetah-analyze:devel/py-cheetah3@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}mako>0:textproc/py-mako@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}requests>0:www/py-requests@${PY_FLAVOR} \
		rst2html:textproc/py-docutils@${PY_FLAVOR} \
		${PYNUMPY} \
		${PY_SETUPTOOLS} \
		${PYTHON_PKGNAMEPREFIX}ruamel.yaml>=0:devel/py-ruamel.yaml@${PY_FLAVOR}

USES=		compiler:c++14-lang cmake ncurses pkgconfig \
		dos2unix python shebangfix
USE_LDCONFIG=	yes
USE_GITHUB=	yes
GH_ACCOUNT=	EttusResearch

BROKEN_FreeBSD_13_aarch64=	fails to configure: Could not find standard set_new_handler function
BROKEN_FreeBSD_13_armv6=	fails to configure: Could not find standard set_new_handler function
BROKEN_FreeBSD_13_armv7=	fails to configure: Could not find standard set_new_handler function

CMAKE_SOURCE_PATH=	${WRKSRC}/host
#CXXFLAGS_amd64=	-msse2 -std=c++11
CXXFLAGS_i386=	-msse2

LDFLAGS+=	${LDFLAGS_${CHOSEN_COMPILER_TYPE}}
LDFLAGS_gcc=	-latomic

SHEBANG_GLOB=	*.py *.py.in *_bist usrp_update*

CMAKE_ARGS+=	-DPKG_LIB_DIR:STRING="share/uhd" -DUHD_TXRX_DEBUG_PRINTS="yes"
# for excruciating debug use this -db
#CMAKE_ARGS+=	--debug-output --trace

OPTIONS_DEFINE=		DOCS DOXYGEN
OPTIONS_DEFAULT=	DOCS

OPTIONS_SUB=		yes

DOXYGEN_BUILD_DEPENDS=	doxygen:devel/doxygen
DOXYGEN_CMAKE_ON=	-DENABLE_DOXYGEN:STRING="ON"
DOXYGEN_CMAKE_OFF=	-DENABLE_DOXYGEN:STRING="OFF" \
			-DENABLE_MANUAL:STRING="OFF"
DOXYGEN_PORTDOCS=	doxygen

#
# XXX N.B. uhd compile will fail if the generated flatbuffers
# files are out of date with current version of flatbuffers.
# There is likely a better way of doing this, but this works.
pre-configure:
.for f in cal_metadata.fbs iq_cal.fbs dsa_cal.fbs pwr_cal.fbs
	(cd ${WRKSRC}/host/include/uhd/cal;flatc --cpp ${f})
.endfor

do-install:
# install host component
# hack the install prefix now
	@${REINPLACE_CMD} -e 's|"${LOCALBASE}"|"${STAGEDIR}${PREFIX}"|g' \
		${CONFIGURE_WRKSRC}/cmake_install.cmake
	cd ${CONFIGURE_WRKSRC} && ${MAKE_CMD} install

post-install:
	${STRIP_CMD} ${STAGEDIR}${DATADIR}/utils/b2xx_fx3_utils
	${STRIP_CMD} ${STAGEDIR}${DATADIR}/utils/usrp_burn_mb_eeprom
	${STRIP_CMD} ${STAGEDIR}${DATADIR}/utils/usrp_burn_db_eeprom
	${STRIP_CMD} ${STAGEDIR}${DATADIR}/utils/query_gpsdo_sensors
	${STRIP_CMD} ${STAGEDIR}${DATADIR}/utils/octoclock_burn_eeprom
	${STRIP_CMD} ${STAGEDIR}${DATADIR}/utils/latency/responder
	${STRIP_CMD} ${STAGEDIR}${DATADIR}/utils/fx2_init_eeprom
	${STRIP_CMD} ${STAGEDIR}${DATADIR}/utils/converter_benchmark

.include <bsd.port.mk>
