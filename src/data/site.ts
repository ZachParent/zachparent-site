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
	resumeLabel: 'View Resume',
	codeLabel: '// NOT YOUR TYPICAL ENGINEER.',
	measurements: {
		vertical: '720',
		horizontal: '1280',
	},
	signoff: ['AGI is coming.', 'Let\'s ensure it benefits everyone'],
	code: {
		name: 'Zach Parent',
		current_role: 'Forward Deployed Engineer',
		focus: [
			'systems design',
			'artificial intelligence',
			'customer success',
		],
		principles: {
			open_source: true,
			impact: true,
		},
	},
} as const;

export const workItems = [
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
		title: "Master's Degree in AI",
		primary: {
			label: 'UPC (BarcelonaTech)',
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
		primary: {
			label: 'Video platform made just for skaters',
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
			icon: 'cube',
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
		label: 'PST (UTC-8)',
		icon: 'clock',
	},
	{
		label: 'Available for new opportunities',
		icon: 'opportunity',
	},
	{
		label: 'zachparent at duck dot com',
		icon: 'mail',
	},
] as const;

export const sidebarGroups = [
	{
		label: 'Focus',
		items: ['Systems', 'Clarity', 'Impact'],
	},
	{
		label: 'Approach',
		items: ['Design', 'Build', 'Iterate'],
	},
] as const;
