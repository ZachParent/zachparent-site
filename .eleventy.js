module.exports = config => {
  const markdownIt = new require('markdown-it')({
    html: true,
    typographer: true,
    linkify: true,
  });

  const markdownItAnchor = require('markdown-it-anchor');
  const markdownItKatex = require('@iktakahiro/markdown-it-katex');
  markdownIt.use(markdownItAnchor);
  markdownIt.use(markdownItKatex);

  config.setLibrary('md', markdownIt);

  config.addPlugin(require('eleventy-plugin-nesting-toc'), {
    tags: ['h3', 'h4', 'h5'],
    ul: false
  });

  config.addPlugin(require('@11ty/eleventy-img').eleventyImageTransformPlugin, {
		// which file extensions to process
		// extensions: "html",
    // svgShortCircuit: true,


		// Add any other Image utility options here:

		// optional, output image formats
		// formats: ["webp", "jpeg"],
		formats: ["auto"],
    // urlPath: '/images/',
    // outputDir: './images/',

		// optional, output image widths
		widths: ["auto"],

		// optional, attributes assigned on <img> override these values.
		// defaultAttributes: {
		// 	loading: "lazy",
		// 	decoding: "async",
		// 	sizes: "100vw",
		// },
	});
  config.addPlugin(require('@11ty/eleventy-plugin-syntaxhighlight'));
  config.addPlugin(require("@11ty/eleventy-plugin-rss"));

  config.addFilter('dateDisplay', require('./filters/date-display.js'));

  config.addPassthroughCopy({ public: './' });

  config.setBrowserSyncConfig({
    files: ['dist/**/*'],
    open: true,
  });

  config.setDataDeepMerge(true);

  config.addCollection('publishedPosts', (collection) =>
    [...collection.getFilteredByGlob('src/posts/*.md')].filter(
      (post) => post.data.published
    )
  );

  return {
    pathPrefix: require('./src/globals/site.json').baseUrl,
    dir: {
      input: 'src',
      output: 'dist',
      includes: 'includes',
      layouts: 'includes/layouts',
      data: 'globals',
    },
  };
};
