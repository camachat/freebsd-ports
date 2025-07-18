PORTNAME=	tuckr
DISTVERSION=	0.11.2
PORTREVISION=	2
CATEGORIES=	sysutils

MAINTAINER=	yuri@FreeBSD.org
COMMENT=	Super-powered replacement for GNU Stow
WWW=		https://github.com/RaphGL/Tuckr

LICENSE=	GPLv3
LICENSE_FILE=	${WRKSRC}/LICENSE

USES=		cargo

USE_GITHUB=	yes
GH_ACCOUNT=	RaphGL
GH_PROJECT=	Tuckr

CARGO_CRATES=	aead-0.5.2 \
		ahash-0.8.12 \
		aho-corasick-1.1.3 \
		ansi-str-0.9.0 \
		ansitok-0.3.0 \
		anstream-0.6.18 \
		anstyle-1.0.10 \
		anstyle-parse-0.2.6 \
		anstyle-query-1.1.2 \
		anstyle-wincon-3.0.7 \
		arc-swap-1.7.1 \
		arrayvec-0.7.6 \
		base62-2.2.1 \
		bitflags-1.3.2 \
		bitflags-2.9.0 \
		block-buffer-0.10.4 \
		bstr-1.12.0 \
		bytecount-0.6.8 \
		cfg-if-1.0.0 \
		chacha20-0.9.1 \
		chacha20poly1305-0.10.1 \
		cipher-0.4.4 \
		clap-4.5.38 \
		clap_builder-4.5.38 \
		clap_derive-4.5.32 \
		clap_lex-0.7.4 \
		colorchoice-1.0.3 \
		cpufeatures-0.2.17 \
		crossbeam-deque-0.8.6 \
		crossbeam-epoch-0.9.18 \
		crossbeam-utils-0.8.21 \
		crypto-common-0.1.6 \
		digest-0.10.7 \
		dirs-6.0.0 \
		dirs-sys-0.5.0 \
		either-1.15.0 \
		enumflags2-0.7.11 \
		enumflags2_derive-0.7.11 \
		equivalent-1.0.2 \
		generic-array-0.14.7 \
		getrandom-0.2.16 \
		getrandom-0.3.3 \
		glob-0.3.2 \
		globset-0.4.16 \
		globwalk-0.8.1 \
		hashbrown-0.15.3 \
		heck-0.5.0 \
		ignore-0.4.23 \
		indexmap-2.9.0 \
		inout-0.1.4 \
		is_terminal_polyfill-1.70.1 \
		itertools-0.11.0 \
		itoa-1.0.15 \
		lazy_static-1.5.0 \
		libc-0.2.172 \
		libredox-0.1.3 \
		log-0.4.27 \
		memchr-2.7.4 \
		minimal-lexical-0.2.1 \
		nom-7.1.3 \
		normpath-1.3.0 \
		once_cell-1.21.3 \
		opaque-debug-0.3.1 \
		option-ext-0.2.0 \
		owo-colors-4.2.0 \
		papergrid-0.15.0 \
		poly1305-0.8.0 \
		proc-macro-error-attr2-2.0.0 \
		proc-macro-error2-2.0.1 \
		proc-macro2-1.0.95 \
		quote-1.0.40 \
		r-efi-5.2.0 \
		rand_core-0.6.4 \
		redox_users-0.5.0 \
		regex-1.11.1 \
		regex-automata-0.4.9 \
		regex-syntax-0.8.5 \
		rpassword-7.4.0 \
		rtoolbox-0.0.3 \
		rust-i18n-3.1.4 \
		rust-i18n-macro-3.1.4 \
		rust-i18n-support-3.1.4 \
		rustversion-1.0.20 \
		ryu-1.0.20 \
		same-file-1.0.6 \
		serde-1.0.219 \
		serde_derive-1.0.219 \
		serde_json-1.0.140 \
		serde_spanned-0.6.8 \
		serde_yaml-0.9.34+deprecated \
		sha2-0.10.9 \
		siphasher-1.0.1 \
		smallvec-1.15.0 \
		stable_deref_trait-1.2.0 \
		strsim-0.11.1 \
		subtle-2.6.1 \
		syn-2.0.101 \
		sys-locale-0.3.2 \
		tabled-0.19.0 \
		tabled_derive-0.11.0 \
		testing_table-0.3.0 \
		thiserror-2.0.12 \
		thiserror-impl-2.0.12 \
		toml-0.8.22 \
		toml_datetime-0.6.9 \
		toml_edit-0.22.26 \
		toml_write-0.1.1 \
		triomphe-0.1.14 \
		typenum-1.18.0 \
		unicode-ident-1.0.18 \
		unicode-width-0.2.0 \
		universal-hash-0.5.1 \
		unsafe-libyaml-0.2.11 \
		utf8parse-0.2.2 \
		version_check-0.9.5 \
		vte-0.14.1 \
		walkdir-2.5.0 \
		wasi-0.11.0+wasi-snapshot-preview1 \
		wasi-0.14.2+wasi-0.2.4 \
		winapi-util-0.1.9 \
		windows-sys-0.52.0 \
		windows-sys-0.59.0 \
		windows-targets-0.52.6 \
		windows_aarch64_gnullvm-0.52.6 \
		windows_aarch64_msvc-0.52.6 \
		windows_i686_gnu-0.52.6 \
		windows_i686_gnullvm-0.52.6 \
		windows_i686_msvc-0.52.6 \
		windows_x86_64_gnu-0.52.6 \
		windows_x86_64_gnullvm-0.52.6 \
		windows_x86_64_msvc-0.52.6 \
		winnow-0.7.10 \
		wit-bindgen-rt-0.39.0 \
		zerocopy-0.8.25 \
		zerocopy-derive-0.8.25 \
		zeroize-1.8.1

PLIST_FILES=	bin/${PORTNAME}

.include <bsd.port.mk>
