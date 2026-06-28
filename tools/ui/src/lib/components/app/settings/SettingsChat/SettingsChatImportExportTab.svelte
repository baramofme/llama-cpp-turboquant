<script lang="ts">
	import type { Component } from 'svelte';
	import { Download, Upload, Trash2 } from '@lucide/svelte';
<<<<<<<< HEAD:tools/ui/src/lib/components/app/settings/SettingsChat/ChatSettingsImportExportTab.svelte
	import { Button, type ButtonVariant } from '$lib/components/ui/button';
	import { DialogConversationSelection, DialogConfirmation } from '$lib/components/app';
========
	import {
		DialogConversationSelection,
		DialogConfirmation,
		DialogExportSettings
	} from '$lib/components/app';
>>>>>>>> upstream/master:tools/ui/src/lib/components/app/settings/SettingsChat/SettingsChatImportExportTab.svelte
	import { createMessageCountMap } from '$lib/utils';
	import { settingsStore } from '$lib/stores/settings.svelte';
	import { conversationsStore, conversations } from '$lib/stores/conversations.svelte';
	import { toast } from 'svelte-sonner';
	import { fade } from 'svelte/transition';
	import { ConversationSelectionMode, HtmlInputType, FileExtensionText } from '$lib/enums';
	import SettingsChatImportExportSection from './SettingsChatImportExportSection.svelte';
	import SettingsGroup from '$lib/components/app/settings/SettingsGroup.svelte';

	let exportedConversations = $state<DatabaseConversation[]>([]);
	let importedConversations = $state<DatabaseConversation[]>([]);
	let showExportSummary = $state(false);
	let showImportSummary = $state(false);

	let showExportDialog = $state(false);
	let showImportDialog = $state(false);
	let availableConversations = $state<DatabaseConversation[]>([]);
	let messageCountMap = $state<Map<string, number>>(new Map());
	let fullImportData = $state<Array<{ conv: DatabaseConversation; messages: DatabaseMessage[] }>>(
		[]
	);

	// Delete functionality state
	let showDeleteDialog = $state(false);

	// Settings import/export state
	let showSettingsExportSummary = $state(false);
	let showSettingsImportSummary = $state(false);
	let showSettingsExportDialog = $state(false);
	let includeSensitiveData = $state(false);

	function handleSettingsExport() {
		showSettingsExportDialog = true;
		includeSensitiveData = false;
	}

	function handleSettingsExportConfirm() {
		showSettingsExportDialog = false;

		try {
			const data = settingsStore.exportSettings(includeSensitiveData);
			const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = `llama_settings_${new Date().toISOString().split('T')[0]}.json`;
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
			URL.revokeObjectURL(url);

			showSettingsExportSummary = true;
			showSettingsImportSummary = false;
			toast.success('Settings exported');
		} catch (err) {
			console.error('Failed to export settings:', err);
			toast.error('Failed to export settings');
		}
	}

	function handleSettingsExportCancel() {
		showSettingsExportDialog = false;
	}

	function handleSettingsImport() {
		try {
			const input = document.createElement('input');
			input.type = HtmlInputType.FILE;
			input.accept = FileExtensionText.JSON;

			input.onchange = async (e) => {
				const file = (e.target as HTMLInputElement)?.files?.[0];
				if (!file) return;

				try {
					const text = await file.text();
					const data = JSON.parse(text);

					if (!data || typeof data !== 'object' || !data.config) {
						toast.error('Invalid settings file: missing config');
						return;
					}

					settingsStore.importSettings(data);

					showSettingsImportSummary = true;
					showSettingsExportSummary = false;
					toast.success('Settings imported successfully');
				} catch (err) {
					console.error('Failed to import settings:', err);
					toast.error('Failed to import settings');
				}
			};

			input.click();
		} catch (err) {
			console.error('Failed to open file picker:', err);
			toast.error('Failed to open file picker');
		}
	}

	async function handleExportClick() {
		try {
			const allConversations = conversations();
			if (allConversations.length === 0) {
				toast.info('No conversations to export');
				return;
			}

			const conversationsWithMessages = await Promise.all(
				allConversations.map(async (conv: DatabaseConversation) => {
					const messages = await conversationsStore.getConversationMessages(conv.id);
					return { conv, messages };
				})
			);

			messageCountMap = createMessageCountMap(conversationsWithMessages);
			availableConversations = allConversations;
			showExportDialog = true;
		} catch (err) {
			console.error('Failed to load conversations:', err);
			alert('Failed to load conversations');
		}
	}

	async function handleExportConfirm(selectedConversations: DatabaseConversation[]) {
		try {
			const allData: ExportedConversation[] = await Promise.all(
				selectedConversations.map(async (conv) => {
					const messages = await conversationsStore.getConversationMessages(conv.id);
					return { conv: $state.snapshot(conv), messages: $state.snapshot(messages) };
				})
			);

			if (allData.length === 1) {
				conversationsStore.downloadConversationFile(allData[0]);
			} else {
				conversationsStore.downloadConversationsArchive(allData);
			}

			exportedConversations = selectedConversations;
			showExportSummary = true;
			showImportSummary = false;
			showExportDialog = false;
		} catch (err) {
			console.error('Export failed:', err);
			alert('Failed to export conversations');
		}
	}

	async function handleImportClick() {
		try {
			const input = document.createElement('input');

			input.type = HtmlInputType.FILE;
			input.accept = `${FileExtensionText.JSON},${FileExtensionText.JSONL},${FileExtensionText.ZIP}`;

			input.onchange = async (e) => {
				const file = (e.target as HTMLInputElement)?.files?.[0];
				if (!file) return;

				try {
					const importedData = await conversationsStore.parseImportFile(file);

					if (importedData.length === 0) {
						throw new Error('No conversations found in file');
					}

					fullImportData = importedData;
					availableConversations = importedData.map((item) => item.conv);
					messageCountMap = createMessageCountMap(importedData);
					showImportDialog = true;
				} catch (err: unknown) {
					const message = err instanceof Error ? err.message : 'Unknown error';

					console.error('Failed to parse file:', err);
					alert(`Failed to parse file: ${message}`);
				}
			};

			input.click();
		} catch (err) {
			console.error('Import failed:', err);
			alert('Failed to import conversations');
		}
	}

	async function handleImportConfirm(selectedConversations: DatabaseConversation[]) {
		try {
			const selectedIds = new Set(selectedConversations.map((c) => c.id));
			const selectedData = $state
				.snapshot(fullImportData)
				.filter((item) => selectedIds.has(item.conv.id));

			await conversationsStore.importConversationsData(selectedData);

			importedConversations = selectedConversations;
			showImportSummary = true;
			showExportSummary = false;
			showImportDialog = false;
		} catch (err) {
			console.error('Import failed:', err);
			alert('Failed to import conversations. Please check the file format.');
		}
	}

	async function handleDeleteAllClick() {
		try {
			const allConversations = conversations();

			if (allConversations.length === 0) {
				toast.info('No conversations to delete');
				return;
			}

			showDeleteDialog = true;
		} catch (err) {
			console.error('Failed to load conversations for deletion:', err);
			toast.error('Failed to load conversations');
		}
	}

	async function handleDeleteAllConfirm() {
		try {
			await conversationsStore.deleteAll();

			showDeleteDialog = false;
		} catch (err) {
			console.error('Failed to delete conversations:', err);
		}
	}

	function handleDeleteAllCancel() {
		showDeleteDialog = false;
	}
</script>

<div class="space-y-12" in:fade={{ duration: 150 }}>
	<SettingsGroup title="Conversations">
		<SettingsChatImportExportSection
			title="Export"
			description="Download your conversations as a ZIP of JSONL files. This includes all messages, attachments, and conversation history."
			IconComponent={Download}
			buttonText="Export conversations"
			onclick={handleExportClick}
			summary={{ show: showExportSummary, verb: 'Exported', items: exportedConversations }}
		/>

		<SettingsChatImportExportSection
			title="Import"
			description="Import one or more conversations from a previously exported ZIP or JSONL file. This will merge with your existing conversations."
			IconComponent={Upload}
			buttonText="Import conversations"
			onclick={handleImportClick}
			summary={{ show: showImportSummary, verb: 'Imported', items: importedConversations }}
		/>

		<SettingsChatImportExportSection
			title="Delete All"
			description="Permanently delete all conversations and their messages. This action cannot be undone. Consider exporting your conversations first if you want to keep a backup."
			IconComponent={Trash2}
			buttonText="Delete all conversations"
			onclick={handleDeleteAllClick}
			titleClass="text-destructive"
			buttonVariant="destructive"
			buttonClass="text-destructive-foreground justify-start justify-self-start bg-destructive hover:bg-destructive/80 md:w-auto"
		/>
	</SettingsGroup>

	<SettingsGroup title="Settings">
		<SettingsChatImportExportSection
			title="Export"
			description="Export your chat settings and preferences as a JSON file."
			IconComponent={Download}
			buttonText="Export settings"
			onclick={handleSettingsExport}
			summary={{ show: showSettingsExportSummary, verb: 'Exported', items: [] }}
		/>

<<<<<<<< HEAD:tools/ui/src/lib/components/app/settings/SettingsChat/ChatSettingsImportExportTab.svelte
			{#if showExportSummary && exportedConversations.length > 0}
				<div class="mt-4 grid overflow-x-auto rounded-lg border border-border/50 bg-muted/30 p-4">
					<h5 class="mb-2 text-sm font-medium">
						Exported {exportedConversations.length} conversation{exportedConversations.length === 1
							? ''
							: 's'}
					</h5>

					<ul class="space-y-1 text-sm text-muted-foreground">
						{#each exportedConversations.slice(0, 10) as conv (conv.id)}
							<li class="truncate">• {conv.name || 'Untitled conversation'}</li>
						{/each}

						{#if exportedConversations.length > 10}
							<li class="italic">
								... and {exportedConversations.length - 10} more
							</li>
						{/if}
					</ul>
				</div>
			{/if}
		</div>

{#snippet section(
	title: string,
	description: string,
	IconComponent: Component,
	buttonText: string,
	onclick: () => void,
	opts: SectionOpts
)}
	{@const buttonClass = opts?.buttonClass ?? 'justify-start justify-self-start md:w-auto'}
	{@const buttonVariant = opts?.buttonVariant ?? 'outline'}
	<div class="grid gap-1 {opts?.wrapperClass ?? ''}">
		<h4 class="mt-0 mb-2 text-sm font-medium {opts?.titleClass ?? ''}">{title}</h4>

			<p class="mb-4 text-sm text-muted-foreground">
				Import one or more conversations from a previously exported JSON file. This will merge with
				your existing conversations.
			</p>

		<Button class={buttonClass} {onclick} variant={buttonVariant}>
			<IconComponent class="mr-2 h-4 w-4" />

			{#if showImportSummary && importedConversations.length > 0}
				<div class="mt-4 grid overflow-x-auto rounded-lg border border-border/50 bg-muted/30 p-4">
					<h5 class="mb-2 text-sm font-medium">
						Imported {importedConversations.length} conversation{importedConversations.length === 1
							? ''
							: 's'}
					</h5>

					<ul class="space-y-1 text-sm text-muted-foreground">
						{#each importedConversations.slice(0, 10) as conv (conv.id)}
							<li class="truncate">• {conv.name || 'Untitled conversation'}</li>
						{/each}

<div class="space-y-6" in:fade={{ duration: 150 }}>
	<div class="space-y-6">
		{@render section(
			'Export Conversations',
			'Download all your conversations as a JSON file. This includes all messages, attachments, and conversation history.',
			Download,
			'Export conversations',
			handleExportClick,
			{ summary: { show: showExportSummary, verb: 'Exported', items: exportedConversations } }
		)}

			<Button
				class="text-destructive-foreground w-full justify-start justify-self-start bg-destructive hover:bg-destructive/80 md:w-auto"
				onclick={handleDeleteAllClick}
				variant="destructive"
			>
				<Trash2 class="mr-2 h-4 w-4" />

				Delete all conversations
			</Button>
		</div>
	</div>
========
		<SettingsChatImportExportSection
			title="Import"
			description="Import chat settings from a previously exported JSON file. This will merge with your existing settings."
			IconComponent={Upload}
			buttonText="Import settings"
			onclick={handleSettingsImport}
			summary={{ show: showSettingsImportSummary, verb: 'Imported', items: [] }}
		/>
	</SettingsGroup>
>>>>>>>> upstream/master:tools/ui/src/lib/components/app/settings/SettingsChat/SettingsChatImportExportTab.svelte
</div>

<DialogExportSettings
	bind:open={showSettingsExportDialog}
	bind:includeSensitiveData
	onConfirm={handleSettingsExportConfirm}
	onCancel={handleSettingsExportCancel}
/>

<DialogConversationSelection
	conversations={availableConversations}
	{messageCountMap}
	mode={ConversationSelectionMode.EXPORT}
	bind:open={showExportDialog}
	onCancel={() => (showExportDialog = false)}
	onConfirm={handleExportConfirm}
/>

<DialogConversationSelection
	conversations={availableConversations}
	{messageCountMap}
	mode={ConversationSelectionMode.IMPORT}
	bind:open={showImportDialog}
	onCancel={() => (showImportDialog = false)}
	onConfirm={handleImportConfirm}
/>

<DialogConfirmation
	bind:open={showDeleteDialog}
	title="Delete all conversations"
	description="Are you sure you want to delete all conversations? This action cannot be undone and will permanently remove all your conversations and messages."
	confirmText="Delete All"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleDeleteAllConfirm}
	onCancel={handleDeleteAllCancel}
/>
