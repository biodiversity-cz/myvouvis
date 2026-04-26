import Head from "next/head";

const MainHead = () => {
  return (
    <Head>
      <title>Quick Diff</title>
      <link
        rel="apple-touch-icon"
        sizes="180x180"
        href="/favicon_io/apple-touch-icon.png"
      />
      <link
        rel="icon"
        type="image/png"
        sizes="32x32"
        href="/favicon_io/favicon-32x32.png"
      />
      <link
        rel="icon"
        type="image/png"
        sizes="16x16"
        href="/favicon_io/favicon-16x16.png"
      />
      <link rel="manifest" href="/favicon_io/manifest.json" />
      <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
      <meta
        name="viewport"
        content="width=device-width, initial-scale=1, minimum-scale=1"
      />
      <meta
        name="description"
        content="Quick Diff is a privacy-first web app that compares two input texts and finds differences. Support diff with different file formats, better Syntax Highlighter, and various diffing modes, such as Side by Side, and Inline mode."
      />
      <meta name="keywords" content="compare text, online diff, diff, diff tool, diff json, diff text, quick diff, quickdiff, online diff tool, vscode, monaco diff editor"></meta>
      <meta name="author" content="admin@proxyman.io" />
      <meta name="msapplication-TileColor" content="#ffffff" />
      <meta name="robots" content="index,follow"></meta>
      <meta name="googlebot" content="index,follow"></meta>
      <meta property="og:site_name" content="QuickDiff"></meta>
      <meta property="og:type" content="website" />
      <meta property="og:locale" content="en_US"></meta>
      <meta property="og:url" content="https://quick-diff.com/" />
      <meta
        property="og:title"
        content="Quick Diff"
      />
      <meta
        property="og:description"
        content="Quick Diff is a privacy-first web app that compares two input texts and finds differences. Support diff with different file formats, better Syntax Highlighter, and various diffing modes, such as Side by Side, and Inline mode."
      />
      <meta property="og:image:width" content="2760"></meta>
      <meta property="og:image:height" content="1500"></meta>
      <meta
        property="og:image"
        content="https://quick-diff.com/quick-diff-thumbnail.png"
      />
      <meta httpEquiv="Content-Type" content="text/html; charset=UTF-8"></meta>
      <meta httpEquiv="Content-Language" content="en"></meta>
      <meta name="apple-mobile-web-app-title" content="QuickDiff"></meta>
      <meta name="application-name" content="QuickDiff"></meta>
      <meta name="theme-color" content="#000000"></meta>
    </Head>
  );
};

export default MainHead;
