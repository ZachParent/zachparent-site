export type IconName =
	| 'file'
	| 'user'
	| 'linkedin'
	| 'map'
	| 'clock'
	| 'mail'
	| 'opportunity'
	| 'code'
	| 'calendar'
	| 'cap'
	| 'cube'
	| 'tag';

type WorkAccent = 'blue' | 'amber' | 'green' | 'neutral';

type WorkPill = {
	label: string;
	icon: IconName;
	accent: WorkAccent;
};

export type WorkItem = {
	number: string;
	title: string;
	primary: WorkPill;
	secondary: WorkPill;
	href?: string;
	external?: boolean;
};

export const navItems = [
	{
		label: 'LinkedIn',
		href: 'https://www.linkedin.com/in/zachparent/',
		icon: 'linkedin',
	},
] as const;

export const homeContent = {
	logo: 'ZP',
	role: 'Forward Deployed Engineer',
	name: 'Zach Parent',
	intro: 'I build software systems that matter to people.',
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
		number: '01',
		title: 'OpenAI',
		primary: {
			label: 'Forward Deployed Engineer',
			icon: 'code',
			accent: 'blue',
		},
		secondary: {
			label: '2025-Present',
			icon: 'calendar',
			accent: 'neutral',
		},
	},
	{
		number: '02',
		title: "UPC (Barcelona Tech)",
		primary: {
			label: 'Master\'s Degree in AI',
			icon: 'cap',
			accent: 'amber',
		},
		secondary: {
			label: '2024-2025',
			icon: 'calendar',
			accent: 'neutral',
		},
	},
	{
		number: '03',
		title: 'Skatefolio',
		href: 'https://skatefolio.com',
		external: true,
		primary: {
			label: 'Skate video platform',
			icon: 'cube',
			accent: 'green',
		},
		secondary: {
			label: '2022-2024',
			icon: 'calendar',
			accent: 'neutral',
		},
	},
	{
		number: '04',
		title: 'Google',
		primary: {
			label: 'Software Engineer',
			icon: 'code',
			accent: 'blue',
		},
		secondary: {
			label: '2019-2022',
			icon: 'calendar',
			accent: 'neutral',
		},
	},
	{
		number: '05',
		title: 'University of Michigan',
		primary: {
			label: 'Aerospace Engineering and Computer Science',
			icon: 'cap',
			accent: 'amber',
		},
		secondary: {
			label: '2015-2019',
			icon: 'calendar',
			accent: 'neutral',
		},
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
