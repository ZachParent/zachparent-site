export type IconName =
	| 'file'
	| 'user'
	| 'github'
	| 'linkedin'
	| 'external'
	| 'map'
	| 'clock'
	| 'mail'
	| 'opportunity'
	| 'code'
	| 'calendar'
	| 'cap'
	| 'cube'
	| 'tag';

export type WorkItem = {
	period: string;
	title: string;
	role: string;
	summary: string;
	href?: string;
	external?: boolean;
};

export const navItems = [
	{
		label: 'GitHub',
		href: 'https://github.com/ZachParent',
		icon: 'github',
		external: true,
	},
	{
		label: 'LinkedIn',
		href: 'https://www.linkedin.com/in/zachparent/',
		icon: 'linkedin',
		external: true,
	},
] as const;

export const homeContent = {
	logo: 'ZP',
	role: 'Forward Deployed Engineer',
	name: 'Zach Parent',
	intro: 'I build production software and applied AI systems with a bias toward real users, clear interfaces, and reliable execution.',
	location: 'San Francisco, CA',
	email: 'zachparent at duck dot com',
	codeLabel: '// NOT YOUR TYPICAL ENGINEER.',
	measurements: {
		vertical: '720',
		horizontal: '1280',
	},
	signoff: ['Let\'s ensure AI benefits everyone'],
	code: {
		name: 'Zach Parent',
		current_role: 'Forward Deployed Engineer',
		focus: [
			'systems design',
			'artificial intelligence',
			'customer success',
		],
	},
} as const;

// Add href, and external when needed, to make a row clickable without changing its visual style.
export const workItems: WorkItem[] = [
	{
		period: '2025-present',
		title: 'OpenAI',
		role: 'Forward Deployed Engineer',
		summary: 'Building with customers at the edge of applied AI and production software.',
	},
	{
		period: '2024-2025',
		title: "UPC (Barcelona Tech)",
		role: 'Master\'s Degree in AI',
		summary: 'Graduate study in artificial intelligence and machine learning systems.',
	},
	{
		period: '2022-2024',
		title: 'Skatefolio',
		href: 'https://skatefolio.com',
		external: true,
		role: 'Founder / Builder',
		summary: 'Created a skate video platform for organizing, sharing, and revisiting clips.',
	},
	{
		period: '2019-2022',
		title: 'Google',
		role: 'Software Engineer',
		summary: 'Worked on reliable, large-scale production systems.',
	},
	{
		period: '2015-2019',
		title: 'University of Michigan',
		role: 'Aerospace Engineering and Computer Science',
		summary: 'Studied engineering fundamentals, software, and systems thinking.',
	},
] as const;

export const contactItems = [
	{
		label: 'San Francisco, CA',
		icon: 'map',
	},
	{
		label: 'zachparent at duck dot com',
		icon: 'mail',
	},
] as const;

export const sidebarGroups: { label: string; items: string[] }[] = [];
