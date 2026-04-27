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

export const codeLines = [
	'<span class="token keyword">const</span> developer = {',
	'  name: <span class="token string">"Zach Parent"</span>,',
	'  role: <span class="token string">"Software Developer"</span>,',
	'  focus: [',
	'    <span class="token string">"systems design"</span>,',
	'    <span class="token string">"developer experience"</span>,',
	'    <span class="token string">"reliable software"</span>,',
	'  ],',
	'  principles: {',
	'    clarity: <span class="token boolean">true</span>,',
	'    simplicity: <span class="token boolean">true</span>,',
	'    impact: <span class="token boolean">true</span>,',
	'  }',
	'};',
	'<span class="token keyword">export default</span> developer;',
] as const;

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
