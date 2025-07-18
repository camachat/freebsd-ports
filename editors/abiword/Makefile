PORTNAME=	abiword
PORTVERSION=	3.0.5
PORTREVISION=	14
CATEGORIES=	editors
MASTER_SITES=	https://dev.alpinelinux.org/archive/${PORTNAME}/
DIST_SUBDIR=	AbiWord

MAINTAINER=	gnome@FreeBSD.org
COMMENT=	Open-source, cross-platform WYSIWYG word processor
WWW=		https://gitlab.gnome.org/World/AbiWord

LICENSE=	GPLv2
LICENSE_FILE=	${WRKSRC}/COPYING

LIB_DEPENDS=	libboost_system.so:devel/boost-libs \
		libpopt.so:devel/popt \
		libpng.so:graphics/png \
		libwv.so:textproc/wv \
		libfribidi.so:converters/fribidi \
		libgoffice-0.10.so:devel/goffice
RUN_DEPENDS=	${LOCALBASE}/share/fonts/dejavu/DejaVuSerif.ttf:x11-fonts/dejavu

USES=		compiler:c++11-lang cpe desktop-file-utils gmake gnome iconv jpeg \
		libtool localbase:ldflags pathfix pkgconfig
CPE_VENDOR=	abisource
USE_GNOME=	libxml2 gtk30 librsvg2 libxslt
USE_LDCONFIG=	yes

GNU_CONFIGURE=	yes
GNU_CONFIGURE_MANPREFIX=${PREFIX}/share
CONFIGURE_ARGS=	--without-gnomevfs --enable-plugins="${ABIWORD_PLUGINS}" \
		--enable-shared="yes" --enable-static="no"

INSTALL_TARGET=	install-strip

PLIST_SUB+=	ABIVERSION=3.0

OPTIONS_DEFINE=	CLIPART DEBUG MATHVIEW NLS PSION \
		TEMPLATES WP
OPTIONS_DEFAULT=CLIPART COLSUGAR SPELLCHECK TEMPLATES
OPTIONS_GROUP=	DICT COLLAB
OPTIONS_GROUP_COLLAB=	COLFAKE COLSERVICE COLSIP COLSUGAR COLTCP COLTELEPATHY COLXMPP
OPTIONS_GROUP_DICT=	GRAMMAR SPELLCHECK
OPTIONS_SUB=	yes

CLIPART_DESC=		Include clipart
COLFAKE_DESC=		Fake collaboration for debugging purposes only
COLLAB_DESC=		Collaboration backend
COLSERVICE_DESC=	Collaborate with AbiCollab.net webservice
COLSIP_DESC=		Collaborate with the SIP/SIMPLE backend
COLSUGAR_DESC=		Collaborate with the Sugar/OLPC environment
COLTCP_DESC=		Collaborate with direct TCP peer to peer connections
COLTELEPATHY_DESC=	Collaborate with the Telepathy backend (experimental)
COLXMPP_DESC=		Collaborate with the Jabber/XMPP backend
DICT_DESC=		Spelling and grammar checking
GRAMMAR_DESC=		Grammar checking plugin
MATHVIEW_DESC=		MathML render plugin
PSION_DESC=		Psion 5(MX) files convertion plugin
SPELLCHECK_DESC=	Spell checking support
TEMPLATES_DESC=		Include additional templates
WP_DESC=		WordPerfect img/doc import/export plugin

# check configure for ALL_PLUGINS for complete list
ABIWORD_PLUGINS=	garble loadbindings opml xslfo bmp iscii sdw babelfish \
			paint command latex pdf wikipedia hrtext openxml \
			goffice wmf kword mif wml s5 freetranslation gdict mht \
			applix t602 docbook google eml hancom presentation \
			clarisworks pdb urldict openwriter mswrite gimp \
			passepartout eml

# opendocument enabled by default, rsvg plugin isn't needed on unix.
# Plugins we don't support:
# gda

CLIPART_CONFIGURE_ENABLE=	clipart

COLFAKE_CONFIGURE_ENABLE=	collab-backend-fake

COLSERVICE_BROKEN=		Does not build with asio from system
COLSERVICE_BUILD_DEPENDS=	${LOCALBASE}/include/asio.hpp:net/asio
COLSERVICE_LIB_DEPENDS=		libsoup-2.4.so:devel/libsoup \
				libgnutls.so:security/gnutls
COLSERVICE_RUN_DEPENDS=		${LOCALBASE}/include/asio.hpp:net/asio
COLSERVICE_CONFIGURE_ENABLE=	collab-backend-service

COLSIP_CONFIGURE_ENABLE=	collab-backend-sip

COLSUGAR_LIB_DEPENDS=		libdbus-glib-1.so:devel/dbus-glib
COLSUGAR_CONFIGURE_ENABLE=	collab-backend-sugar

COLTCP_BROKEN=			Does not build with asio from system
COLTCP_BUILD_DEPENDS=		${LOCALBASE}/include/asio.hpp:net/asio
COLTCP_RUN_DEPENDS=		${LOCALBASE}/include/asio.hpp:net/asio
COLTCP_CONFIGURE_ENABLE=	collab-backend-tcp

COLTELEPATHY_LIB_DEPENDS=	libtelepathy-glib.so:net-im/telepathy-glib
COLTELEPATHY_CONFIGURE_ENABLE=	collab-backend-telepathy

COLXMPP_LIB_DEPENDS=		libloudmouth-1.so:net-im/loudmouth
COLXMPP_CONFIGURE_ENABLE=	collab-backend-xmpp

DEBUG_CONFIGURE_ENABLE=		debug

GRAMMAR_IMPLIES=		SPELLCHECK
GRAMMAR_LIB_DEPENDS=		liblink-grammar.so:textproc/link-grammar
GRAMMAR_VARS=			ABIWORD_PLUGINS+=grammar

MATHVIEW_LIB_DEPENDS=		libmathview.so:x11-toolkits/gtkmathview
MATHVIEW_USES=			bison
MATHVIEW_VARS=			ABIWORD_PLUGINS+=mathview

NLS_USES=			gettext
NLS_CONFIGURE_ENABLE=		nls

PSION_LIB_DEPENDS=		libpsiconv.so:converters/psiconv
PSION_VARS=			ABIWORD_PLUGINS+=psion

SPELLCHECK_LIB_DEPENDS=		libenchant.so:textproc/enchant
SPELLCHECK_CONFIGURE_ENABLE=	spell

TEMPLATES_CONFIGURE_ENABLE=	templates

WP_LIB_DEPENDS=	libwpg-0.3.so:graphics/libwpg03 \
		libwpd-0.10.so:textproc/libwpd010 \
		libwps-0.4.so:textproc/libwps
WP_USE=		gnome=libgsf
WP_VARS=	ABIWORD_PLUGINS+="wpg wordperfect"

.include <bsd.port.options.mk>

.if ${PORT_OPTIONS:MCOL*}
ABIWORD_PLUGINS+=	collab
.if ${PORT_OPTIONS:MCOLFAKE} || ${PORT_OPTIONS:MCOLSERVICE} || ${PORT_OPTIONS:MCOLSIP} || ${PORT_OPTIONS:MCOLTCP} || ${PORT_OPTIONS:MCOLTELEPATHY} || ${PORT_OPTIONS:MCOLXMPP}
PLIST_SUB+=		COLLAB=""
.else
PLIST_SUB+=		COLLAB="@comment "
.endif
.endif

post-patch:
	@${REINPLACE_CMD} -e 's|glib/.*\.h>|glib.h>|g' \
		${WRKSRC}/goffice-bits/goffice/app/goffice-app.h \
		${WRKSRC}/src/af/util/xp/ut_go_file.h
# The following substitutions ensure that plugins are linked with libabiword
# from the work or stage directory instead of a previously installed library.
# In pseudo code:
# s,PLUGIN_LIBS="$PLUGIN_LIBS -Lsrc -labiword",
#   PLUGIN_LIBS="src/libabiword.la $PLUGIN_LIBS",
# s,foo_LIBS="$foo_LIBS $PLUGIN_LIBS",
#   foo_LIBS="$PLUGIN_LIBS $foo_LIBS",
# The initial value of PLUGIN_LIBS may contain -L/usr/local/lib so it would
# have to come after -Lsrc, but -Lsrc is not correct when libtool relinks
# plugins during staging so use src/libabiword.la and let libtool figure out
# what flags are needed.  Secondly, switch around foo_LIBS and PLUGIN_LIBS
# because foo_LIBS may contain -L/usr/local/lib.
	@${REINPLACE_CMD} \
		-e "s,\$$PLUGIN_LIBS \"'-L\$${top_builddir}/src'\" -labiword-\$$ABIWORD_SERIES,\\\\\$${top_builddir}/src/libabiword-\$$ABIWORD_SERIES.la \$$PLUGIN_LIBS," \
		-e "/_LIBS=/s,\"\\(.*\\)\"\\('\$${PLUGIN_LIBS}.*'\\),\\2\" \\1\"," \
		-e 's|" == "|" = "|g' ${WRKSRC}/configure

.include <bsd.port.mk>
