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
		label: 'Resume',
		href: '#resume',
		icon: 'file',
		external: false,
	},
	{
		label: 'About',
		href: '#about',
		icon: 'user',
		external: false,
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
	role: 'Software Developer',
	name: 'Zach Parent',
	intro: 'I build reliable, scalable software that solves real problems and empowers people.',
	resumeLabel: 'View Resume',
	codeLabel: '// SYSTEMS. CODE. IMPACT.',
	measurements: {
		vertical: '720',
		horizontal: '1280',
	},
	signoff: ['Built with intention', 'Shipped with care'],
	code: {
		name: 'Zach Parent',
		role: 'Software Developer',
		focus: ['systems design', 'developer experience', 'reliable software'],
		principles: {
			clarity: true,
			simplicity: true,
			impact: true,
		},
	},
} as const;

export const workItems = [
	{
		number: '01',
		title: 'OpenAI',
		primary: {
			label: 'Forward Deploy Engineer',
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
		title: "Master's Degree",
		primary: {
			label: 'Barcelona / AI',
			icon: 'cap',
			accent: 'amber',
		},
		secondary: {
			label: 'AI',
			icon: 'tag',
			accent: 'neutral',
		},
	},
	{
		number: '03',
		title: 'Skatefolio',
		primary: {
			label: 'Project for skaters',
			icon: 'cube',
			accent: 'green',
		},
		secondary: {
			label: 'Product',
			icon: 'tag',
			accent: 'neutral',
		},
	},
] as const;

export const contactItems = [
	{
		label: 'Boston, MA',
		icon: 'map',
	},
	{
		label: 'ET (UTC-4)',
		icon: 'clock',
	},
	{
		label: 'Available for new opportunities',
		icon: 'opportunity',
	},
	{
		label: 'zach.parent.dev@gmail.com',
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
