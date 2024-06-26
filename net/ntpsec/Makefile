PORTNAME=	ntpsec
DISTVERSIONPREFIX=	NTPsec_
DISTVERSION=	1_2_3
CATEGORIES=	net

MAINTAINER=	yuri@FreeBSD.org
COMMENT=	Network Time Protocol suite, refactored
WWW=		https://gitlab.com/NTPsec/ntpsec/

LICENSE=	BSD2CLAUSE

LIB_DEPENDS=	libdns_sd.so:net/mDNSResponder

USES=		cpe bison gettext-runtime pkgconfig python shebangfix ssl waf
USE_GITHUB=	yes
USE_LDCONFIG=	${PREFIX}/lib/ntp

SHEBANG_FILES=	${WRKSRC}/ntpclients/ntp*.py
python_OLD_CMD=	@PYSHEBANG@

CONFLICTS=	ntp ntp-devel openntpd

MAKE_ENV=	NOSYNC=1 # workaround for the console problem: https://gitlab.com/NTPsec/ntpsec/issues/599

TEST_TARGET=	check

OPTIONS_DEFINE=			DOCS REFCLOCKS
OPTIONS_DEFAULT=		REFCLOCKS

DOCS_CONFIGURE_ENABLE=	doc
DOCS_BUILD_DEPENDS=	asciidoctor:textproc/rubygem-asciidoctor # alternatively ntpsec can use asciidoc:textproc/asciidoc but it prefers asciidoctor

REFCLOCKS_DESC=	Enable all refclocks
REFCLOCKS_VARS=	CONFIGURE_ARGS+=--refclock=all

PORTSCOUT=	limit:^\d+\. # prevent alpha-numeric strings

post-install:
	# autoplist: plist varies depending on the python version
	@cd ${STAGEDIR}${PREFIX} && \
		${FIND} * -type f -or -type l | ${SED} -e 's|^share/man/.*|&.gz|' >> ${TMPPLIST}
	# strip binaries
	@${STRIP_CMD} \
		${STAGEDIR}${PREFIX}/bin/ntpfrob \
		${STAGEDIR}${PREFIX}/bin/ntptime \
		${STAGEDIR}${PREFIX}/lib/libntpc.so.1.1.0 \
		${STAGEDIR}${PREFIX}/sbin/ntpd

.include <bsd.port.mk>
