#!/usr/bin/env python3
"""CPU/process fixtures; never signal actual experiment processes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import torch
import stop_at_epoch as m
from periodic_evaluate import prepare_job, settings, committed_run


class StopFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='epoch-cap-fixture-')
        self.root=Path(self.temp.name);self.run=self.root/'result'/'test';self.run.mkdir(parents=True)
        self.processes=[]
        trainer=self.run/'runtime/code/train_distributed.py';trainer.parent.mkdir(parents=True)
        trainer.write_text('import subprocess,sys,time\np=subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"])\ntime.sleep(120)\n')
        self.config={'epochs':30,'early_stopping_patience':0,'output':str(self.run),
                     'source_sha256':{'code/train.py':'fixture'}}
        self.config_path=self.root/'config.json';m.write_json(self.config_path,self.config)
        m.write_json(self.run/'run_config.json',self.config)
        self.trainer=subprocess.Popen([sys.executable,str(trainer),'--config',str(self.config_path)],start_new_session=True)
        self.processes.append(self.trainer)
        self.evaluator=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True)
        self.processes.append(self.evaluator)
        m.write_json(self.run/'launch.json',{'pid':self.trainer.pid,'config':str(self.config_path)})
        self.binding=m.bind_run(self.run,2)
        self.identity={'val':{'image_count':1601,'unique_source_count':1601,'partial':False,'split':'val'}}
        self.publish(1)
        deadline=time.monotonic()+3
        while not m.child_identities(self.trainer.pid) and time.monotonic()<deadline:time.sleep(.02)
        self.children=m.child_identities(self.trainer.pid)
        assert self.children

    def tearDown(self):
        for identity in self.children+[m.process_identity(p.pid) for p in self.processes]:
            if identity:
                m.send(identity,9);m.send(identity,18)
        for p in self.processes:
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:pass
        self.temp.cleanup()

    def publish(self,completed,last_epoch=None):
        rows=[];val=None
        for e in range(1,completed+1):
            val={'epoch':e,'image_count':1601,'validation_unique_source_count':1601,'partial':False,
                 'checkpoint_selection_uses_test':False,'data_identity':self.identity['val'],'miou':.5+e*.1}
            m.write_json(self.run/f'validation_epoch_{e:03d}.json',val)
            rows.append({'epoch':e,'validation_miou':val['miou'],'best_epoch':e,'best_miou':val['miou'],
                         'partial':False,'checkpoint_selection_uses_test':False})
        (self.run/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.summary={'format':'lusnar_vit_run_v1','completed_epochs':completed,'best_epoch':completed,
                      'best_miou':val['miou'],'partial':False,'validation_images':1601,
                      'checkpoint_selection_uses_test':False,'official_test_used_for_selection':False,
                      'data_identity':self.identity,'model_config':{'variant':'fixture'},'status':'training'}
        m.write_json(self.run/'summary.json',self.summary)
        saved={'format':'lusnar_vit_checkpoint_v1','epoch':last_epoch or completed,'partial':False,
               'train_config':self.config,'data_identity':self.identity,'model_config':self.summary['model_config'],
               'best_epoch':completed,'best_miou':val['miou'],'metrics':val,'model':{'fake':torch.tensor([completed])}}
        torch.save(saved,self.run/'last.pt');torch.save(saved,self.run/'best.pt')

    def test_waits_then_stops_exact_tree_preserves_best_and_final(self):
        before=m.sha256(self.run/'run_config.json')
        self.assertEqual(m.tick(self.binding)['status'],'watching')
        self.assertTrue(m.alive(self.binding['trainer_identity']))
        self.publish(2)
        best=m.sha256(self.run/'best.pt');last=m.sha256(self.run/'last.pt')
        self.assertEqual(m.tick(self.binding)['status'],'complete')
        self.assertFalse(m.alive(self.binding['trainer_identity']))
        self.assertFalse(any(m.alive(p) for p in self.children))
        self.assertIsNone(self.evaluator.poll())
        self.assertEqual(m.sha256(self.run/'best.pt'),best)
        self.assertEqual(m.sha256(self.run/'final.pt'),last)
        self.assertEqual(m.sha256(self.run/'run_config.json'),before)
        summary,config,rows=committed_run(self.run)
        self.assertEqual(summary['status'],'complete');self.assertEqual(summary['completed_epochs'],2)
        self.assertEqual(m.tick(self.binding)['status'],'complete')

    def test_pid_identity_mismatch_never_signals(self):
        self.publish(2)
        self.binding['trainer_identity']['start_ticks']='wrong'
        with self.assertRaisesRegex(ValueError,'identity'):m.tick(self.binding)
        self.assertIsNone(self.trainer.poll());self.assertIsNone(self.evaluator.poll())
        self.assertFalse((self.run/'final.pt').exists())

    def test_bad_checkpoint_releases_pause_and_does_not_stop(self):
        self.publish(2,last_epoch=1)
        with self.assertRaisesRegex(ValueError,'last.pt'):m.tick(self.binding)
        self.assertIsNone(self.trainer.poll());self.assertTrue(m.alive(self.children[0]))
        state=Path(f'/proc/{self.trainer.pid}/stat').read_text().rsplit(')',1)[1].split()[0]
        self.assertNotEqual(state,'T')
        self.assertFalse((self.run/'final.pt').exists())

    def test_uncommitted_history_does_not_stop(self):
        self.publish(2)
        summary=dict(self.summary,completed_epochs=1,best_epoch=1,best_miou=.6)
        m.write_json(self.run/'summary.json',summary)
        self.assertEqual(m.tick(self.binding)['status'],'watching')
        self.assertIsNone(self.trainer.poll())

    def test_changed_run_config_never_signals(self):
        self.publish(2);m.write_json(self.run/'run_config.json',{**self.config,'epochs':99})
        with self.assertRaisesRegex(ValueError,'configuration'):m.tick(self.binding)
        self.assertIsNone(self.trainer.poll())


if __name__=='__main__':unittest.main()
