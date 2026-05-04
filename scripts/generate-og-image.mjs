import { existsSync, unlinkSync } from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import process from 'node:process';

const args = new Set(process.argv.slice(2));
const shouldOpen = args.has('--open');
const shouldBuild = !args.has('--no-build');
const allowExistingPreview = args.has('--use-existing-preview') || process.env.OG_IMAGE_USE_EXISTING_PREVIEW === '1';
const width = Number(process.env.OG_IMAGE_WIDTH || '1200');
const height = Number(process.env.OG_IMAGE_HEIGHT || '630');
const chromeHeightOffset = Number(process.env.OG_IMAGE_CHROME_HEIGHT_OFFSET || '87');
const chromeWindowHeight = Number(process.env.OG_IMAGE_CHROME_WINDOW_HEIGHT || String(height + chromeHeightOffset));
const port = Number(process.env.OG_IMAGE_PORT || '4321');
const host = process.env.OG_IMAGE_HOST || '127.0.0.1';
const imagePath = path.resolve(process.cwd(), process.env.OG_IMAGE_OUTPUT || 'public/og-image.png');
const screenshotPath = chromeWindowHeight > height ? `${imagePath}.tmp.png` : imagePath;
const pagePath = process.env.OG_IMAGE_PATH || '/og-image';
const queryParams = new URLSearchParams({
	width: String(width),
	height: String(height),
});
const previewUrl = `http://${host}:${port}${pagePath}?${queryParams.toString()}`;
const openImage = process.env.OG_IMAGE_OPEN === '1' || shouldOpen;

const chromeCandidates = [
	process.env.CHROME_PATH,
	'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
	'/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
	'/Applications/Chromium.app/Contents/MacOS/Chromium',
	'/usr/bin/google-chrome',
	'/usr/bin/google-chrome-stable',
	'/usr/bin/chromium-browser',
	'/usr/bin/chromium',
];
const magickCandidates = [
	process.env.MAGICK_PATH,
	'/opt/homebrew/bin/magick',
	'/usr/local/bin/magick',
	'/usr/bin/magick',
];

const resolveChrome = () => chromeCandidates.find((candidate) => candidate && existsSync(candidate));
const chromePath = resolveChrome();
const resolveMagick = () => magickCandidates.find((candidate) => candidate && existsSync(candidate));

if (!chromePath) {
	console.error('[og-image] Chrome executable not found. Set CHROME_PATH to a valid browser binary.');
	process.exit(1);
}

const waitForServer = async () => {
	const start = Date.now();
	const timeoutMs = 30000;

	while (Date.now() - start < timeoutMs) {
		try {
			const response = await fetch(previewUrl, { method: 'GET', signal: AbortSignal.timeout(500) });
			if (response.ok) {
				return true;
			}
		} catch {
			// Retry until the server is ready.
		}

		await new Promise((resolve) => {
			setTimeout(resolve, 250);
		});
	}

	throw new Error(`Timed out waiting for ${previewUrl}`);
};

const checkPortBusy = async () => {
	try {
		await fetch(`http://${host}:${port}`, {
			method: 'GET',
			signal: AbortSignal.timeout(300),
		});
		return true;
	} catch {
		return false;
	}
};

const runBuild = async () => new Promise((resolve, reject) => {
	const build = spawn('npm', ['run', 'build:no-og'], {
		stdio: 'inherit',
		env: {
			...process.env,
			NODE_ENV: process.env.NODE_ENV || 'production',
		},
	});

	build.on('error', reject);
	build.on('close', (code) => {
		if (code === 0) {
			resolve();
			return;
		}
		reject(new Error(`Build exited with code ${code}`));
	});
});

const startPreview = async () => {
	if (await checkPortBusy()) {
		if (!allowExistingPreview) {
			throw new Error(`Port ${port} is already in use. Set OG_IMAGE_PORT to a free port or pass --use-existing-preview to reuse a running server at ${previewUrl}.`);
		}

		await waitForServer();
		return null;
	}

	const preview = spawn('npm', ['run', 'preview', '--', '--host', host, '--port', String(port)], {
		stdio: 'inherit',
		env: {
			...process.env,
			NODE_ENV: process.env.NODE_ENV || 'production',
		},
	});

	await waitForServer();
	return preview;
};

const runChromeScreenshot = () => new Promise((resolve, reject) => {
	const chrome = spawn(
		chromePath,
		[
			'--headless=new',
			'--no-sandbox',
			'--disable-gpu',
			'--hide-scrollbars',
			`--window-size=${width},${chromeWindowHeight}`,
			`--screenshot=${screenshotPath}`,
			previewUrl,
		],
		{ stdio: 'inherit' },
	);

	chrome.on('error', reject);
	chrome.on('close', (code) => {
		if (code === 0) {
			resolve();
			return;
		}
		reject(new Error(`Chrome exited with code ${code}`));
	});
});

const cropScreenshot = () => {
	if (screenshotPath === imagePath) {
		return Promise.resolve();
	}

	const magickPath = resolveMagick();

	if (!magickPath) {
		throw new Error('ImageMagick not found. Install ImageMagick or set MAGICK_PATH so the oversized Chrome screenshot can be cropped.');
	}

	return new Promise((resolve, reject) => {
		const crop = spawn(
			magickPath,
			[
				screenshotPath,
				'-crop',
				`${width}x${height}+0+0`,
				'+repage',
				imagePath,
			],
			{ stdio: 'inherit' },
		);

		crop.on('error', reject);
		crop.on('close', (code) => {
			try {
				unlinkSync(screenshotPath);
			} catch {
				// Best-effort cleanup only.
			}

			if (code === 0) {
				resolve();
				return;
			}

			reject(new Error(`ImageMagick exited with code ${code}`));
		});
	});
};

const openImageFile = () => {
	if (!openImage) {
		return;
	}

	const openCommand = process.platform === 'darwin'
		? 'open'
		: process.platform === 'win32'
			? 'start'
			: 'xdg-open';
	spawn(openCommand, [imagePath], { stdio: 'ignore', detached: true, shell: true }).unref();
};

let previewProcess;

try {
	if (shouldBuild) {
		await runBuild();
	}

	previewProcess = await startPreview();
	await runChromeScreenshot();
	await cropScreenshot();
	openImageFile();
	console.log(`[og-image] Generated ${imagePath}`);
	console.log(`[og-image] OG render source: ${previewUrl}`);
} catch (error) {
	console.error('[og-image] Failed to generate OG image:', error.message);
	process.exit(1);
} finally {
	if (previewProcess && !previewProcess.killed) {
		previewProcess.kill('SIGINT');
	}
}
