PORTNAME=	lives
DISTVERSION=	3.2.0
PORTREVISION=	24
CATEGORIES=	multimedia
MASTER_SITES=	http://lives-video.com/releases/
DISTNAME=	LiVES-${PORTVERSION}

MAINTAINER=	vvd@FreeBSD.org
COMMENT=	Video editing system
WWW=		http://lives-video.com/

LICENSE=		GPLv3+ LGPL3+
LICENSE_COMB=		multi
LICENSE_FILE_GPLv3+ =	${WRKSRC}/COPYING
LICENSE_FILE_LGPL3+ =	${WRKSRC}/COPYING.LGPL

BUILD_DEPENDS=	${LOCALBASE}/include/linux/input.h:devel/evdev-proto \
		analyseplugin:audio/ladspa
LIB_DEPENDS=	liboil-0.3.so:devel/liboil \
		libpng.so:graphics/png \
		libmjpegutils.so:multimedia/mjpegtools
RUN_DEPENDS=	analyseplugin:audio/ladspa \
		ffprobe:multimedia/ffmpeg \
		sox:audio/sox

USES=		autoreconf compiler:c++11-lib gettext-runtime gettext-tools \
		gl gnome libtool localbase pathfix perl5 pkgconfig python \
		sdl shebangfix xorg
USE_GL=		gl glu
USE_GNOME=	cairo gdkpixbuf gtk30 intltool
USE_LDCONFIG=	yes
USE_SDL=	sdl sdl2
USE_XORG=	x11 xrender

SHEBANG_FILES=	build-lives-rfx-plugin build-lives-rfx-plugin-multi smogrify \
		tools/autolives.pl lives-plugins/plugins/encoders/*_encoder \
		lives-plugins/marcos-encoders/*encoder* \
		lives-plugins/plugins/encoders/multi_encoder3 \
		lives-plugins/plugins/playback/audiostream/audiostreamer.pl
GNU_CONFIGURE=	yes
CONFIGURE_ARGS=	--disable-ldvgrab --disable-prctl --disable-system-weed --disable-dirac
INSTALL_TARGET=	install-strip
WRKSRC=		${WRKDIR}/${PORTNAME}-${PORTVERSION}

DOC_FILES=	AUTHORS COPYING COPYING.LGPL ChangeLog FEATURES GETTING.STARTED NEWS README \
		OMC/lives-OMC.txt RFX/LiVES-Perl.odt RFX/rfxbuilder.odt \
		RFX/RFX.spec docs/c++-bindings.odt docs/clip_format.txt \
		lives-plugins/marcos-encoders/README.multi_encoder \
		weed-docs/weedspec.txt weed-docs/weedevents.txt

OPTIONS_DEFINE=	ALSA DOCS DOXYGEN DV FFMPEG FFTW FREI0R IMAGEMAGICK JACK \
		LIBEXPLAIN LIBVISUAL NLS OGG OPENCV OPTIMIZED_CFLAGS \
		PROJECTM PULSEAUDIO THEORA V4L WAYLAND
OPTIONS_DEFAULT=	CDDA2WAV DV FFTW GHOSTSCRIPT IMAGEMAGICK JACK \
			LAME MPG123 MPLAYER NLS OGG OGG123 OGM THEORA \
			X264 XDG_SCREENSAVER XDOTOOL XWININFO YTDLP

OPTIONS_GROUP=			RUNTIME EXT_WIN_CAPTURE MPLAYERS MP3
OPTIONS_GROUP_EXT_WIN_CAPTURE=	WMCTRL XDOTOOL
OPTIONS_GROUP_MP3=		MPG123 MPG321
OPTIONS_GROUP_MPLAYERS=		MPLAYER MPV
OPTIONS_GROUP_RUNTIME=		CDDA2WAV GHOSTSCRIPT LAME MATROSKA MENCODER OGG123 OGM \
				SSWF X264 XDG_SCREENSAVER XWININFO YTDLP
OPTIONS_SUB=			yes

EXT_WIN_CAPTURE_DESC=		Allows external window capture via xdotool or wmctrl
LIBEXPLAIN_DESC=		Used to explain Unix and Linux system call errors
MENCODER_DESC=			Use mencoder as video file and movie encoder
MP3_DESC=			Runtime check order is: mpg321, mpg123, mplayer, mplayer2, mpv
MPG321_DESC=			MP3 decoding support via mpg321
MPLAYERS_DESC=			Install mplayer/mpv (runtime check)
MPV_DESC=			MPV media player support
OGG123_DESC=			Use ogg123 from vorbis-tools
OGM_DESC=			Use ogmtools for manipulating with OGM files
PROJECTM_DESC=			projectM support
RUNTIME_DESC=			Runtime detected utilities
SSWF_DESC=			Use sswf for manipulating with Flash movies
WMCTRL_DESC=			Install wmctrl
XDG_SCREENSAVER_DESC=		Used to disable screensaver during playback
XDOTOOL_DESC=			Install xdotool
XWININFO_DESC=			Used for external window capture
YTDLP_DESC=			Install yt-dlp (youtube-dl fork) - requires SYMLINK option

ALSA_LIB_DEPENDS=		libasound.so:audio/alsa-lib
ALSA_CONFIGURE_OFF=		--disable-alsa

CDDA2WAV_RUN_DEPENDS=		cdda2wav:sysutils/cdrtools

DOXYGEN_BUILD_DEPENDS=		doxygen:devel/doxygen
DOXYGEN_CONFIGURE_OFF=		--disable-doxygen

DV_LIB_DEPENDS=			libdv.so:multimedia/libdv
DV_RUN_DEPENDS=			encodedv:multimedia/libdv
DV_CONFIGURE_OFF=		--disable-libdv

FFMPEG_LIB_DEPENDS=		libavcodec.so:multimedia/ffmpeg
FFMPEG_CONFIGURE_OFF=		--disable-ffmpeg

FFTW_LIB_DEPENDS=		libfftw3.so:math/fftw3 \
				libfftw3f.so:math/fftw3-float
FFTW_CONFIGURE_OFF=		--disable-fftw3f

FREI0R_BUILD_DEPENDS=		frei0r>0:graphics/frei0r
FREI0R_RUN_DEPENDS=		frei0r>0:graphics/frei0r
FREI0R_CONFIGURE_OFF=		--disable-frei0r

GHOSTSCRIPT_USES=		ghostscript:run

IMAGEMAGICK_USES=		magick:run

JACK_LIB_DEPENDS=		libjack.so:audio/jack \
				libjackserver.so:audio/jack
JACK_CONFIGURE_OFF=		--disable-jack

LAME_RUN_DEPENDS=		lame:audio/lame

LIBEXPLAIN_LIB_DEPENDS=		libexplain.so:devel/libexplain
LIBEXPLAIN_CONFIGURE_OFF=	--disable-libexplain

LIBVISUAL_LIB_DEPENDS=		libvisual-0.4.so:graphics/libvisual04
LIBVISUAL_CONFIGURE_OFF=	--disable-libvisual

MATROSKA_RUN_DEPENDS=		mkvmerge:multimedia/mkvtoolnix

MENCODER_RUN_DEPENDS=		mencoder:multimedia/mencoder

MPG123_RUN_DEPENDS=		mpg123:audio/mpg123

MPG321_RUN_DEPENDS=		mpg321:audio/mpg321

MPLAYER_RUN_DEPENDS=		mplayer:multimedia/mplayer

MPV_RUN_DEPENDS=		mpv:multimedia/mpv

NLS_USES=			gettext-tools

OGG123_RUN_DEPENDS=		ogg123:audio/vorbis-tools

OGG_LIB_DEPENDS=		libogg.so:audio/libogg
OGG_CONFIGURE_OFF=		--disable-ogg

OGM_RUN_DEPENDS=		ogmmerge:multimedia/ogmtools

OPENCV_LIB_DEPENDS=		libopencv_calib3d.so:graphics/opencv \
				libopencv_core.so:graphics/opencv
OPENCV_CONFIGURE_OFF=		--disable-opencv

OPTIMIZED_CFLAGS_CONFIGURE_ON=	--enable-turbo

PROJECTM_LIB_DEPENDS=		libprojectM.so:graphics/libprojectm
PROJECTM_CONFIGURE_OFF=		--disable-projectM

PULSEAUDIO_LIB_DEPENDS=		libpulse.so:audio/pulseaudio
PULSEAUDIO_CONFIGURE_OFF=	--disable-pulse

SSWF_RUN_DEPENDS=		sswf:graphics/sswf

THEORA_LIB_DEPENDS=		libtheora.so:multimedia/libtheora
THEORA_CONFIGURE_OFF=		--disable-theora

V4L_BUILD_DEPENDS=		${LOCALBASE}/include/linux/videodev2.h:multimedia/v4l_compat
V4L_LIB_DEPENDS=		libv4l2.so:multimedia/libv4l
V4L_CONFIGURE_OFF=		--disable-v4l

WAYLAND_LIB_DEPENDS=		libwayland-client.so:graphics/wayland \
				libwayland-egl.so:graphics/wayland \
				libwayland-cursor.so:graphics/wayland
WAYLAND_CONFIGURE_OFF=		--disable-wayland

WMCTRL_RUN_DEPENDS=		wmctrl:x11/wmctrl-fork

X264_RUN_DEPENDS=		x264:multimedia/x264

XDG_SCREENSAVER_RUN_DEPENDS=	xdg-screensaver:devel/xdg-utils

XDOTOOL_RUN_DEPENDS=		xdotool:x11/xdotool

XWININFO_RUN_DEPENDS=		xwininfo:x11/xwininfo

YTDLP_RUN_DEPENDS=		yt-dlp:www/yt-dlp

post-patch:
	@${FIND} ${WRKSRC} -name Makefile.in | ${XARGS} ${REINPLACE_CMD} \
		's/^LIBS = $$/&@LIBS@/'
	@${REINPLACE_CMD} -e 's|/usr/local/lib/|${LOCALBASE}/lib/|g' \
		${WRKSRC}/lives-plugins/weed-plugins/frei0r.c
	@${REINPLACE_CMD} 's|/usr/bin/perl|${perl_CMD}|' \
		${WRKSRC}/build-lives-rfx-plugin
	@${REINPLACE_CMD} -e '/haar_analyser\.la/d' \
		${WRKSRC}/lives-plugins/weed-plugins/gdk/Makefile.am
# ISO C++17 does not allow 'register' storage class specifier
	@${REINPLACE_CMD} -e 's,register int,int,' \
		${WRKSRC}/lives-plugins/plugins/playback/video/openGL.cpp \
		${WRKSRC}/lives-plugins/weed-plugins/farneback_analyser.cpp \
		${WRKSRC}/lives-plugins/weed-plugins/gdk/haar_analyser.cpp \
		${WRKSRC}/lives-plugins/weed-plugins/projectM.cpp

pre-build-NLS-on:
	@(cd ${WRKSRC}/po && ${DO_MAKE_BUILD} update-po)

post-install-DOCS-on:
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	${INSTALL_DATA} ${DOC_FILES:S|^|${WRKSRC}/|} ${STAGEDIR}${DOCSDIR}

post-install-DOXYGEN-on:
	@(cd ${STAGEDIR}${PREFIX} && ${FIND} ${DOCSDIR_REL}/html \
		-not -type d >> ${TMPPLIST})

.include <bsd.port.mk>
