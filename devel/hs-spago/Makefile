PORTNAME=	spago
DISTVERSION=	0.21.0
PORTREVISION=	1
CATEGORIES=	devel
MASTER_SITES=	https://github.com/purescript/purescript-docs-search/releases/download/v0.0.11/docs-search-app.js?dummy=/:dsa11 \
		https://github.com/purescript/purescript-docs-search/releases/download/v0.0.11/purescript-docs-search?dummy=/:pds11 \
		https://github.com/purescript/purescript-docs-search/releases/download/v0.0.10/docs-search-app.js?dummy=/:dsa10 \
		https://github.com/purescript/purescript-docs-search/releases/download/v0.0.10/purescript-docs-search?dummy=/:pds10
DISTFILES=	docs-search-app-0.0.11.js:dsa11 \
		purescript-docs-search-0.0.11:pds11 \
		docs-search-app-0.0.10.js:dsa10 \
		purescript-docs-search-0.0.10:pds10
EXTRACT_ONLY=	${DISTNAME}${EXTRACT_SUFX}

MAINTAINER=	haskell@FreeBSD.org
COMMENT=	PureScript package manager and build tool
WWW=		https://github.com/spacchetti/spago/

LICENSE=	BSD3CLAUSE
LICENSE_FILE=	${WRKSRC}/LICENSE

BUILD_DEPENDS=	ghc-9.2.8:lang/ghc92
RUN_DEPENDS=	ca_root_nss>0:security/ca_root_nss

USES=		cabal
USE_GITHUB=	yes
GH_ACCOUNT=	purescript
USE_LOCALE=	en_US.UTF-8

USE_CABAL=	Diff-0.4.1 \
		Glob-0.10.2_3 \
		OneTuple-0.3.1_2 \
		QuickCheck-2.14.2 \
		StateVar-1.2.2 \
		adjunctions-4.4.1 \
		aeson-2.0.3.0_1 \
		aeson-better-errors-0.9.1.1_1 \
		aeson-pretty-0.8.9_2 \
		ansi-terminal-0.11.3 \
		ansi-wl-pprint-0.6.9_3 \
		appar-0.1.8 \
		asn1-encoding-0.9.6_2 \
		asn1-parse-0.9.5 \
		asn1-types-0.3.4 \
		assoc-1.0.2_2 \
		async-2.2.4_1 \
		async-pool-0.9.1 \
		atomic-write-0.2.0.7 \
		attoparsec-0.14.4_1 \
		auto-update-0.1.6 \
		base-compat-0.12.1 \
		base-compat-batteries-0.12.1_2 \
		base-orphans-0.8.6 \
		base16-bytestring-1.0.2.0 \
		base64-bytestring-1.2.1.0 \
		basement-0.0.16 \
		basic-prelude-0.7.0 \
		bifunctors-5.5.12 \
		blaze-builder-0.4.2.2_1 \
		bower-json-1.1.0.0 \
		byteorder-1.0.4 \
		cabal-doctest-1.0.9_1 \
		case-insensitive-1.2.1.0 \
		cborg-0.2.7.0_1 \
		cborg-json-0.2.4.0 \
		cereal-0.5.8.2_1 \
		charset-0.3.9 \
		chunked-data-0.3.1 \
		classy-prelude-1.5.0.2 \
		clock-0.8.3 \
		colour-2.3.6 \
		comonad-5.0.8_1 \
		conduit-1.3.4.2 \
		conduit-extra-1.3.6 \
		connection-0.3.1_1 \
		contravariant-1.5.5 \
		cookie-0.4.5 \
		cryptohash-sha256-0.11.102.1 \
		cryptonite-0.30 \
		data-default-class-0.1.2.0 \
		data-fix-0.3.2_2 \
		dhall-1.41.1_3 \
		distributive-0.6.2.1_1 \
		dlist-1.0 \
		dlist-instances-0.1.1.1 \
		dotgen-0.4.3 \
		easy-file-0.2.2 \
		either-5.0.2 \
		fast-logger-3.1.1 \
		fgl-5.7.0.3 \
		file-embed-0.0.15.0 \
		foldl-1.4.12_3 \
		free-5.1.8 \
		fsnotify-0.3.0.1_2 \
		generic-lens-2.2.1.0_1 \
		generic-lens-core-2.2.1.0_1 \
		half-0.3.1 \
		hashable-1.4.0.2 \
		hostname-1.0 \
		hourglass-0.2.12 \
		hsc2hs-0.68.8 \
		http-client-0.7.11_1 \
		http-client-tls-0.3.6.1 \
		http-conduit-2.3.8 \
		http-types-0.12.3 \
		indexed-profunctors-0.1.1 \
		indexed-traversable-0.1.2_1 \
		indexed-traversable-instances-0.1.1_1 \
		integer-logarithms-1.0.3.1_2 \
		invariant-0.5.6 \
		iproute-1.7.12 \
		kan-extensions-5.2.4 \
		keys-3.12.3_2 \
		lens-family-core-2.1.0 \
		libyaml-0.1.4 \
		libyaml-clib-0.2.5 \
		lifted-base-0.2.3.12 \
		managed-1.0.9_1 \
		megaparsec-9.2.1 \
		memory-0.17.0 \
		microlens-0.4.13.0 \
		microlens-mtl-0.2.0.2 \
		mime-types-0.1.0.9 \
		mmorph-1.2.0_2 \
		monad-control-1.0.3.1 \
		monad-logger-0.3.36_2 \
		monad-loops-0.4.3 \
		mono-traversable-1.0.15.3 \
		mono-traversable-instances-0.1.1.0 \
		mtl-compat-0.2.2 \
		mutable-containers-0.3.4 \
		network-3.1.2.7 \
		network-uri-2.6.4.1 \
		old-locale-1.0.0.7_2 \
		old-time-1.1.0.3_2 \
		open-browser-0.2.1.0 \
		optional-args-1.0.2_1 \
		optparse-applicative-0.17.0.0 \
		parser-combinators-1.3.0 \
		parsers-0.12.11 \
		pem-0.2.4 \
		pointed-5.0.4 \
		pretty-simple-4.1.1.0 \
		prettyprinter-1.7.1 \
		prettyprinter-ansi-terminal-1.1.3 \
		primitive-0.7.3.0_2 \
		profunctors-5.6.2_2 \
		random-1.2.1.1 \
		repline-0.4.2.0 \
		resourcet-1.2.5 \
		retry-0.9.2.1 \
		rio-0.1.22.0 \
		rio-orphans-0.1.2.0 \
		safe-0.3.19 \
		safe-exceptions-0.1.7.3 \
		say-0.1.0.1 \
		scientific-0.3.7.0_2 \
		semialign-1.2.0.1_2 \
		semigroupoids-5.3.7 \
		semigroups-0.20 \
		semver-range-0.2.8 \
		serialise-0.2.5.0 \
		socks-0.6.1 \
		split-0.2.3.4_2 \
		splitmix-0.1.0.4 \
		stm-chans-3.0.0.6 \
		streaming-commons-0.2.2.4 \
		strict-0.4.0.1_4 \
		stringsearch-0.3.6.6_1 \
		tagged-0.8.6.1_2 \
		tar-0.5.1.1_5 \
		temporary-1.3 \
		text-manipulate-0.3.1.0 \
		text-short-0.1.5 \
		th-abstraction-0.4.3.0 \
		th-compat-0.1.3_1 \
		th-env-0.1.1 \
		th-lift-0.8.2_1 \
		th-lift-instances-0.1.19 \
		these-1.1.1.1_5 \
		time-compat-1.9.6.1_3 \
		tls-1.6.0 \
		transformers-base-0.4.6 \
		transformers-compat-0.7.1_1 \
		turtle-1.6.1 \
		typed-process-0.2.10.1 \
		unix-compat-0.6 \
		unix-time-0.4.7 \
		unliftio-0.2.22.0 \
		unliftio-core-0.2.0.1_2 \
		unordered-containers-0.2.19.1 \
		uri-encode-1.5.0.7_2 \
		utf8-string-1.0.2 \
		uuid-types-1.0.5_2 \
		vector-0.12.3.1_2 \
		vector-algorithms-0.8.0.4_2 \
		vector-instances-3.4_1 \
		versions-5.0.3 \
		void-0.7.3 \
		with-utf8-1.0.2.3 \
		witherable-0.4.2_3 \
		x509-1.7.7 \
		x509-store-1.6.9 \
		x509-system-1.6.7 \
		x509-validation-1.6.12 \
		yaml-0.11.11.2_2 \
		zlib-0.6.3.0

post-extract:
	${CP} ${DISTDIR}/${DIST_SUBDIR}/docs-search-app-0.0.11.js ${WRKSRC}/templates/
	${CP} ${DISTDIR}/${DIST_SUBDIR}/purescript-docs-search-0.0.11 ${WRKSRC}/templates/
	${CP} ${DISTDIR}/${DIST_SUBDIR}/docs-search-app-0.0.10.js ${WRKSRC}/templates/
	${CP} ${DISTDIR}/${DIST_SUBDIR}/purescript-docs-search-0.0.10 ${WRKSRC}/templates/

.include <bsd.port.mk>
